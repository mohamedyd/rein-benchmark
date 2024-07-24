import difflib
from error_generator.strategies.utilities.inst_checker import Factory
from error_generator.strategies.utilities.inst_checker import Similar_First


class Similar_Based_Active_Domain(object):
    def __init__(self,name="similar_based_active_domain"):
        self.name=name
        self.called=False

    
        
    def run(self,row,col,selected_value,dataset):

        #make columns once
        #each cell of all_temp is one column
        if self.called == False:

            ins_similar=Similar_First()
            f=Factory()
            self.all_temp=f.get(ins_similar,dataset)
            self.called=True


        similar = difflib.get_close_matches(selected_value, self.all_temp[col], n=1000, cutoff=0)
        while selected_value in similar: similar.remove(selected_value)
        if len(similar) == 0:
            # here we need to pic the value that is not similar to selected value because
            # the value that picked was uniqe
            similar = difflib.get_close_matches(selected_value, self.all_temp[col], n=len(dataset), cutoff=0)
            while selected_value in similar: similar.remove(selected_value)


        return similar[0]

