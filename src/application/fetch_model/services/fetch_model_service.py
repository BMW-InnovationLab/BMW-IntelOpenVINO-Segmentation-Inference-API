import os
import json
import uuid


class FetchModelService:
    def __init__(self):
        """
        Sets the models base directory, and initializes some dictionaries.
        Saves the loaded model's hashes to a json file, so the values are saved even though the API went down.
        """
        self.models_dict = {}
        self.path = '/'
        file_name = self.path + 'models_hash/model_hash.json'
        file_exists = os.path.exists(file_name)
        if file_exists:
            try:
                with open(file_name) as json_file:
                    self.models_hash_dict = json.load(json_file)
            except:
                self.models_hash_dict = {}
        else:
            with open(file_name, 'w'):
                self.models_hash_dict = {}
        self.labels_hash_dict = {}
        self.base_models_dir = self.path + 'models'

    def fetch_all_models(self):
        """
        Loads all the available models.
        :return: Returns a List of all models and their respective hashed values
        """
        models = self.list_models()
        for model in models:
            if model not in self.models_hash_dict:
                self.get_uuid(model)
        for key in list(self.models_hash_dict):
            if key not in models:
                del self.models_hash_dict[key]
        with open('/models_hash/model_hash.json', 'w') as fp:
            json.dump(self.models_hash_dict, fp)
        return self.models_hash_dict

    def list_models(self):
        """
        Lists all the available models.
        :return: List of models
        """
        return [folder for folder in os.listdir(self.base_models_dir) if
                os.path.isdir(os.path.join(self.base_models_dir, folder))]

    def get_uuid(self, model):
        """
        get all models in folder saved_model
        :param:
        :return: str of all models available
        """

        for file in os.listdir(self.base_models_dir + "/" + model):
            if file.endswith(".xml"):
                self.models_hash_dict[model] = str(uuid.uuid4())
