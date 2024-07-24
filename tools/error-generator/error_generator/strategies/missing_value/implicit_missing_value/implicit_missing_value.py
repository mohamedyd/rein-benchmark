from difflib import SequenceMatcher
from operator import itemgetter

class Implicit_Missing_Value(object):
    def __init__(self,name="Implicit_Missing_Value"):
        self.name=name
        self.dic={"phone number":"11111111","education":"Some college",
                  "workclass":"?","date":"20010101","Ref_ID":"-1","Regents Num":"s","Junction Control":"-1",
                  "Birthday":"20010101","EVENT_DT":"20030101","state":"Alabama","country":"Afghanistan",
                  "email":"...@gmail.com","ssn":"111111111"}


    def run(self,row,col,selected_value,dataset):

        #insted putting the median and mode for implicit missing value
        #we do label matching and acording the dictionary we replace data

        # similar_first=Similar_First()
        # similar_first.similar_first(dataset)
        #
        # mod_value=similar_first.mod_value
        # median_value=similar_first.median_value
        #
        # col_list = [median_value[col], mod_value[col]]
        #
        #
        # rand = np.random.randint(0, 2)
        # selected = col_list[rand]
        #
        # while str(selected_value) == str(selected):
        #     col_list = col_list.remove(selected)
        #     if col_list is None:
        #         selected = median_value + median_value
        #
        # if (isinstance(selected, list)):
        #     if len(selected) > 1:
        #         selected = selected[0]


        col_name =dataset[0][col]
        all_similarty=[]
        for key in self.dic:

            simirarty=SequenceMatcher(None, key, col_name).ratio()
            all_similarty.append((simirarty,key))

        max_match_key=max(all_similarty, key=itemgetter(0))

        if max_match_key[0]>0.3:
            selected = self.dic[str(max_match_key[1])]
        else:
            selected = "N/A"

        return selected
