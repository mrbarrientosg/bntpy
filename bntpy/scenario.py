import os
from typing import List, Optional
from math import ceil
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators.StructureScore import K2Score
from pgmpy.models.BayesianNetwork import BayesianNetwork
import toml
from bntpy.parameters import Parameters
from networkx.drawing.nx_pydot import to_pydot
from pgmpy.sampling import GibbsSampling, BayesianModelSampling
import multiprocessing
import subprocess
from os import listdir
from os.path import isfile, join


class Scenario:
    def __init__(self, parameters: Parameters, file=None):
        self.target_runner: str = ""
        self.train_instances: List[str] = [""]
        self.parameters: Parameters = parameters
        self.train_instance_dir: Optional[str] = None
        self.population_size: int = 10 * parameters.nb_params
        self.max_iterations: int = 5 * parameters.nb_params
        self.select_size: int = ceil(0.3 * self.population_size)
        self.sample_size: int = int(self.population_size / 2)

        # Variable
        self.last_individual: int = 0
        self.last_instance: int = 0
        self.population = pd.DataFrame(
            columns=["ID", *parameters.get_names(), "FITNESS"])
        self.all_configurations = pd.DataFrame(
            columns=["ID", *parameters.get_names(), "FITNESS"])
        self.elitists = pd.DataFrame(
            columns=["ID", *parameters.get_names(), "FITNESS"])
        self.model = None
        self.read_scenario(file)

    def read_scenario(self, file):
        params = toml.load(file)

        for key in params:
            setattr(self, key, params[key])

        self.train_instances = [join(self.train_instance_dir, f) for f in listdir(self.train_instance_dir) if isfile(join(self.train_instance_dir, f))]

    def initialize_population(self):
        for _ in range(self.population_size):
            row = {
                "ID": self.last_individual,
                "FITNESS": np.random.randint(0, 10000)
            }

            for name in self.parameters.get_names():
                domain = self.parameters.get_domain(name)
                if self.parameters.get_type(name) == "i":
                    row[name] = np.random.randint(
                        domain[0], domain[1] + 1, dtype=int)
                else:
                    row[name] = np.random.randint(0, len(domain), dtype=int)

            self.last_individual += 1
            self.population = self.population.append(
                row, ignore_index=True)

    def reduce_population(self):
        self.population.sort_values("FITNESS", inplace=True)
        self.population = self.population.head(self.select_size)
        self.population = self.population.reset_index(drop=True)

    def run_individual(self, data):
        fitness = 0.0
        for (_id, seed, instance) in data[2]:
            id_individual = data[1]["ID"]
            command = [self.target_runner, str(
                id_individual), str(_id), str(seed), instance]

            for name in self.parameters.get_names():
                command.append(self.parameters.get_switch(
                    name) + str(self.parameters.get_value(name, data[1][name])))

            process = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, universal_newlines=True)

            fitness += float(process.stdout)

        return (data[0], fitness / len(data[2]))

    def callback_individual(self, result):
        for idx, fitness in result:
            self.population.iloc[idx]["FITNESS"] = fitness

    def run_population(self, intances):
        """Aqui se deberian correr todo los individuos"""
        count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=count)
        data = [(idx, row, intances)
                for idx, row in self.population.iterrows()]
        masync = pool.map_async(self.run_individual,
                                data, callback=self.callback_individual)
        masync.wait()

    def create_instances(self):
        seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
        instances = (self.last_individual, seed,
                     self.train_instances[self.last_individual % len(self.train_instances)])
        self.last_individual += 1
        return list([instances])

    def save_all_configurations(self):
        self.all_configurations.sort_values("ID", inplace=True)
        _all_configurations = self.all_configurations.apply(lambda row, names: pd.Series([self.parameters.get_value(
            name, row[name]) for name in names], index=names), axis=1, names=self.parameters.get_names())
        _all_configurations.insert(0, "ID", self.all_configurations["ID"])
        _all_configurations["FITNESS"] = self.all_configurations["FITNESS"]
        _all_configurations.to_csv("all_configurations.csv", index=False)

    def save_elitists_configurations(self):
        self.elitists.sort_values("ID", inplace=True)
        self.elitists.drop_duplicates(subset=["ID"], inplace=True)
        elitists_configurations = self.elitists.apply(lambda row, names: pd.Series([self.parameters.get_value(
            name, row[name]) for name in names], index=names), axis=1, names=self.parameters.get_names())
        elitists_configurations.insert(0, "ID", self.elitists["ID"])
        elitists_configurations["FITNESS"] = self.elitists["FITNESS"]
        elitists_configurations.to_csv("elitists.csv", index=False)

    def run(self):
        if not os.access(self.target_runner, os.X_OK):
            print("Target runner is not an executable file")
            return

        print("# Bayesian Network Tuning Parameter ----------")
        print("# Version: 1.0.0")
        print("# Author: Matias Barrientos")
        print("# --------------------------------------------")
        print("# Scenario Initialization")
        print(f"# Population size: {self.population_size}")
        print(f"# Select individual size: {self.select_size}")
        print(f"# Iterations: {self.max_iterations}")
        print(f"# Sample size: {self.sample_size}")
        print()

        intances = self.create_instances()
        self.initialize_population()
        self.run_population(intances)
        self.all_configurations = self.population
        self.reduce_population()

        for i in range(self.max_iterations):
            print(f"Iteration {i + 1} of {self.max_iterations}")

            # Crear red
            est = HillClimbSearch(self.population[self.parameters.get_names()])

            self.model = BayesianNetwork(est.estimate(
                K2Score(self.population[self.parameters.get_names()]), max_indegree=2))
            self.model.fit(self.population[self.parameters.get_names()])
            self.model.check_model()

            inference = BayesianModelSampling(self.model)
            sample = inference.likelihood_weighted_sample(
                size=self.sample_size)
            sample = sample.drop(columns=["_weight"])
            sample["ID"] = np.arange(
                self.last_individual, self.last_individual + self.sample_size)
            sample["FITNESS"] = np.full(self.sample_size, -1.0)

            self.population = pd.concat(
                [self.population, sample]).reset_index(drop=True)
            intances = self.create_instances()
            self.run_population(intances)
            self.all_configurations = pd.concat([self.all_configurations, self.population[self.population["ID"].isin(
                sample["ID"].to_numpy())]]).reset_index(drop=True)
            self.reduce_population()
            print(
                f"Best-so-far configurations: {self.population.head(3)['ID'].to_numpy()} \t fitness: {self.population.head(3)['FITNESS'].to_numpy()}")
            print("Description of the best-so-far configuration:")
            print(self.population.head(3).apply(lambda row, names: pd.Series([self.parameters.get_value(
                name, row[name]) for name in names], index=names), axis=1, names=self.parameters.get_names()).to_string())
            self.elitists = pd.concat(
                [self.elitists, self.population.head(3)]).reset_index(drop=True)
            print()
            print("Bayesian network representation:")
            print(to_pydot(self.model).to_string())
            print()
        self.save_all_configurations()
        self.save_elitists_configurations()
