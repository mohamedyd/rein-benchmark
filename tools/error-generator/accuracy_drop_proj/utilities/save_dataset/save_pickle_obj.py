import pickle


class Save_Pickle_Obj(object):
    def __init__(self):
        pass

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)





