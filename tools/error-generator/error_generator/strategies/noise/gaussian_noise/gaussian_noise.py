import numpy as np
import chainer, chainer.links as L,chainer.functions as F
from chainer import Variable

class Gaussian_Noise(object):
    def __init__(self,name="Gaussian_Noise"):
        self.name=name
        
        
    def run(self,row,col,selected_value,dataset):
        
        noise_rate=0.7
        if (isinstance(selected_value, str)):

            asci_str = 0
            for ch in selected_value:
                code = ord(ch)
                asci_str = code + asci_str
            added_str = chr(asci_str)

            temp_array = np.array([asci_str]).astype(np.float32)
            # batch,dim = v.data.shape[0],v.data.shape[1]
            batch = 1
            ones = Variable(np.ones((batch), dtype=np.float32))
            new_value = F.gaussian(temp_array, noise_rate * ones)

            if int(new_value.data[0]) == asci_str:
                print("your noise rate is not alot so the strings will reverse as noise")
                new_value = selected_value[::-1]
                replaced_value = new_value

            else:

                rand = np.random.randint(0, len(selected_value))
                replaced_value = selected_value[:rand] + added_str + selected_value[rand:]

        else:
            temp_array = np.array([selected_value]).astype(np.float32)
            # batch,dim = v.data.shape[0],v.data.shape[1]
            batch = 1
            ones = Variable(np.ones((batch), dtype=np.float32))
            new_value = F.gaussian(temp_array, noise_rate * ones)
            replaced_value = new_value.data[0]

        

        return replaced_value
        
        