import random

class Random_Active_Domain(object):
    def __init__(self,name="random_active_domain"):
        self.name=name

    def run(self,row,col,selected_value,dataset):

        col_rand = random.randint(0, len(dataset[0]) - 1)
        row_rand = random.randint(1, len(dataset) - 1)
        
        while (row_rand == row):
            row_rand = random.randint(1, len(dataset) - 1)

        temp_random_method = dataset[row_rand][col]
        
        return temp_random_method