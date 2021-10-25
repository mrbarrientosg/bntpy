from tqdm import tqdm
from functools import partialmethod
from bntpy.parameters import Parameters
from bntpy.scenario import Scenario
import argparse

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


parser = argparse.ArgumentParser(prog="Bayesian Network Tunning", description="Automatic Parameter Configuration")
parser.add_argument("-s", "--scenario", help="Path to scenario file (fomat file .toml)", action="store", type=str, default=None)
parser.add_argument("-p", "--parameters", help="Path to paremeters file (fomat file .toml)", action="store", type=str, required=True)
# parser.add_argument("-i", "--iterations", help="Number of max iterations", action="store", type=int)

args = parser.parse_args()



parameters = Parameters(file_path=args.parameters)
scenario = Scenario(parameters, args.scenario)
scenario.run()
