import random

class Value_Selector(object):
    def __init__(self):
        self.value_selector_history=[]
        self.picked_value=[]
        pass
    def number(self,dataset,percentage):
        number = int((percentage / 100.0) * (len(dataset) ))
        return number
    
    def select_value(self,dataset,number,mute_column):
        for i in range(number):
            random_value = random.randint(1, len(dataset) - 1)
            while random_value in self.value_selector_history:
                random_value = random.randint(1, len(dataset) - 1)
            self.value_selector_history.append(random_value)

            col = random.randint(0, len(dataset[0]) - 1)
            while col in mute_column:
                col = random.randint(0, len(dataset[0]) - 1)

            input_value = dataset[random_value][col]

            while (len(input_value) == 0):
                random_value = random.randint(1, len(dataset) - 1)
                while random_value in self.value_selector_history:
                    random_value = random.randint(1, len(dataset) - 1)
                self.value_selector_history.append(random_value)
                input_value = dataset[random_value][col]
            self.picked_value.append([random_value,col,input_value])
        return self.picked_value

