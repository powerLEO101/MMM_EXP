import pickle

def pickle_save(x, path):
    with open(path, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    with open(path, 'rb') as handle:
        x = pickle.load(handle)
    return x
