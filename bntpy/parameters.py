import toml
import pandas as pd


class Parameters:
    def __init__(self, parameters=None, file_path=None):
        self.parameters = parameters
        self.nb_params: int = 0

        if self.parameters is None:
            self._read_file(file_path)

    def _read_file(self, file_path):
        params = toml.load(file_path)
        self.parameters = pd.DataFrame(
            columns=["name", "switch", "type", "domain"])

        for key in params:
            self.parameters = self.parameters.append(params[key],
                                                     ignore_index=True)

        self.nb_params = self.parameters.shape[0]

    def get_switch(self, name: str):
        return self.parameters.loc[self.parameters["name"] ==
                                   name].iloc[0]["switch"]

    def get_type(self, name: str):
        return self.parameters.loc[self.parameters["name"] ==
                                   name].iloc[0]["type"]

    def get_domain(self, name: str):
        return self.parameters.loc[self.parameters["name"] ==
                                   name].iloc[0]["domain"]

    def get_value(self, name: str, idx):
        if self.get_type(name) == "c":
            return self.parameters.loc[self.parameters["name"] ==
                                       name].iloc[0]["domain"][idx]
        return idx

    def get_names(self):
        return self.parameters["name"].to_list()
