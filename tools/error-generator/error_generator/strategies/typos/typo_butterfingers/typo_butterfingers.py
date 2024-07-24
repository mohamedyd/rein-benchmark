
from error_generator.strategies.typos.typo_butterfingers.butterfingers import butterfinger


class Typo_Butterfingers(object):
    def __init__(self,name="Typo_Butterfingers"):
        self.name=name


    def run(self,row,col,selected_value,dataset):
        temp = butterfinger(selected_value)
        return temp
