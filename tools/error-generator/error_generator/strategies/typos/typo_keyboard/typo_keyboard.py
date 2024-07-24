from random import *
w=choice
o=ord


class Typo_Keyboard(object):
    def __init__(self,name="Typo_Keyboard"):
        self.name=name
        
    def run(self,row,col,selected_value,dataset):
        temp = "".join(w([z] * 0 + [w(["", z * 2] + [chr(o(w(
            "DGRC FHTV GJYB UOK HKUN JLIM KO NK BMJ IPL O WA ETF ADWZ RYG YIJ CBG QES ZCD TUH XS SQ VNH XVF SFEX WRD".split()[
                (o(z) & 31) - 6])) | o(z) & 32)] * z.isalpha())]) for z in selected_value)

        return temp