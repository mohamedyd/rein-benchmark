from error_generator.strategies.utilities.input_output import Read_Write



 

class test(object):
    def __init__(self,name="test"):
        self.name=name
    
    
dataset,dataframe = Read_Write.read_csv_dataset("../datasets/test.csv")
Read_Write.write_csv_dataset("../outputs/address_{}.csv".format(test().name),dataset)
print(dataset)
    
    