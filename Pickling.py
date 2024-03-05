import pickle
class PickleHelper:
    def save_to(self, location, obj):
        with open(location, "wb") as f:
            pickle.dump(obj, f)

    def load_back(self, location):
        with open(location, 'rb') as f:
            return pickle.load(f)
