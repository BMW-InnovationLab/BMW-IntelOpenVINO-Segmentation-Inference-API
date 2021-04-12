import os
import json
import uuid


class FetchPalette:
    def __init__(self):
        """
        Sets the models base directory, and initializes some dictionaries.
        Saves the loaded model's hashes to a json file, so the values are saved even though the API went down.
        """
        self.path = '/'
        self.palette = []
        self.base_models_dir = self.path + 'models'

    def get_palette(self,model_name):
        """
        Loads the model in case it's not loaded.
        Returns the model's labels.
        :param model_name: Model name
        :return: List of model labels
        """
        try:
            model_path = os.path.join(self.base_models_dir, model_name)
        except Exception as ex:
            raise ex
        #read from configuration.json
        path = os.path.join(model_path,'palette.txt')
        try:
            infile = open((path),'r')
            for line in infile:
                self.palette.append(int(line.strip('\n')))
        except Exception as ex:
            raise ex
        return self.palette
