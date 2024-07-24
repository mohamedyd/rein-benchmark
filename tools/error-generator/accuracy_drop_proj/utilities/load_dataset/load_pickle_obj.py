import pickle

class Load_Pickle_Obj(object):
    def __init__(self):
        pass
    def load(self,file_name):

        infile = open(file_name, 'rb')
        new_obj = pickle.load(infile)
        infile.close()
        return new_obj