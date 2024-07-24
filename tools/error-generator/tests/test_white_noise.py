from error_generator.api.error_generator_api import Error_Generator
from error_generator.strategies.noise.white_noise.white_noise import White_Noise
from error_generator.strategies.utilities.list_selected import List_selected
from error_generator.strategies.utilities.input_output import Read_Write

class Test_White_Noise(object):
    def __init__(self, name="test_white_noise"):
        self.name = name





# ------------------------------- this is test part ----------------------------------


dataset,dataframe = Read_Write.read_csv_dataset("../datasets/test.csv")

mymethod=White_Noise()


myselector=List_selected()


mygen=Error_Generator()
new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=20,dataset=dataset,mute_column=[])


# #create instance of test
inst_test=Test_White_Noise()


#write to output
Read_Write.write_csv_dataset("../outputs/{}.csv".format(inst_test.name), new_dataset)

