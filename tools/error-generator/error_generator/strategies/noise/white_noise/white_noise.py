import numpy as np
import math
import sys
class White_Noise(object):
    def __init__(self,name="White_noise"):
        self.name=name
        
    def run(self,row,col,selected_value,dataset):
        mu, sigma = 0, 1  # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1)

        asci_number = ""
        noisy_value = ""
        if (isinstance(selected_value, str)):

            for ch in selected_value:
                code = ord(ch)
                digits = int(math.log10(code)) + 1
                if digits <= 2:
                    code = str(code)
                    code = code.zfill(3)
                else:
                    code = str(code)
                asci_number = asci_number + code
            # if you pass the really large sentence the python can't handel that so we replace that number
            # with maximum number that python can accept

            if int(asci_number) > sys.maxsize:
                asci_number = int(sys.maxsize)
            string_noise = int(int(asci_number) * noise[0])
            string_noise = string_noise + int(asci_number)
            string_noise = str(string_noise)
            three_number = int(len(string_noise) / 3)
            if len(string_noise) % 3 != 0:
                three_number = three_number + 1
            for i in range(three_number):
                three = string_noise[-3:]
                noisy_value = noisy_value + chr(abs(int(three)))
                string_noise = string_noise.replace(three, '', 1)
            noisy_value = noisy_value[::-1]
            


        else:
            add_value = float(noise[0]) * float(selected_value)
            if (isinstance(selected_value, float)):
                noisy_value = float(selected_value) + add_value
            else:
                if (int(float(selected_value) + add_value)) == selected_value:
                    noisy_value = int(float(selected_value) + add_value) + 1
                noisy_value = int(float(selected_value) + add_value)



        return noisy_value
        
        