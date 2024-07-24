from error_generator.api.error_generator_api import Error_Generator
from error_generator.strategies.utilities.input_output import Read_Write
from error_generator.strategies.utilities.list_selected import List_selected
from error_generator.strategies.switch_value.Similar_based_active_domain.similar_based_active_domain import \
    Similar_Based_Active_Domain


class Test_Similar_Based_Active_Domain(object):
    def __init__(self, name="similar_based_active_domain"):
        self.name = name





# ------------------------------- this is test part ----------------------------------


dataset,dataframe = Read_Write.read_csv_dataset("../datasets/test.csv")

mymethod=Similar_Based_Active_Domain()


myselector=List_selected()


mygen=Error_Generator()
new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=20,dataset=dataset,mute_column=[])


# #create instance of test
inst_test=Test_Similar_Based_Active_Domain()


#write to output
Read_Write.write_csv_dataset("../outputs/{}.csv".format(inst_test.name), new_dataset)



