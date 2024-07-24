import heapq
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination

"""
this  method first sort the rows according to uncetainity and features according to information gain and then pick
first rows that classifer is not sure about them and feature are more likely to change 
algorithm 3
"""
class Change_Uncertainty_Rankfeatures(object):
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan):
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))

        used_row ={}
        occurred_change = 0
        all_changed = 1
        change_done = False
        x_train_changed = np.copy(x_train)

        #---------------------find the order of the feature according to information gain-----------------------

        model = ExtraTreesClassifier()
        model.fit(x_train, y_train)

        print("combinatio of feature")

        information_gain = {}
        for i in range(len(model.feature_importances_)):
            information_gain.update({i: model.feature_importances_[i]})

        print(information_gain)
        ranked_information_dic = {}
        sum_gain = 0
        for L in range(0,x_train.shape[1] + 1 ):
            for subset in Change_Combination.combinations_index(self,information_gain.keys(), L):
                if not subset:
                    pass
                else:
                    print(subset)
                    for key in subset:
                        sum_gain = sum_gain + information_gain.get(key)
                    ranked_information_dic.update({tuple(subset): sum_gain})
                    sum_gain = 0

        print("create all subset")

        all_subset = sorted(ranked_information_dic.items(), key=lambda item: len(item[0]) * 1000 - item[1], reverse=False)
        print(all_subset)

        #---------------------------finding the order of the row according to uncertainity-------------------------

        probability = mnb.predict_proba(x_train)
        print(probability)

        print("finding uncertainity")

        uncertainty={}
        for index,roww in enumerate(probability):
            largest_val =heapq.nlargest(2, roww)
            uncertainty.update({index:1-(np.abs(np.subtract(largest_val[0],largest_val[1])))})
            largest_val=[]
            # print(index,row,np.subtract(largest_val[0],largest_val[1]))


        #sort the uncertainty
        uncertainty_sorted=sorted(uncertainty.items(), key=lambda x:x[1],reverse=True)
        print(uncertainty_sorted)

        print("changing")
        #---------------------------------------------changing--------------------------------------------

        for i in range(len(change_plan["key"])):
            occurred_change = 0
            #sort the row according to uncertainty

            indices=[]

            for key_dic in uncertainty_sorted:
                if y_train[key_dic[0]] == change_plan["key"][i][0]:
                    indices.append(key_dic[0])

            print(indices)


            #this is normal indices
            # indices_2 = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]


            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))
            print("try in indices")
            for p in range(len(indices)):

                if (all_changed == number_change_requested + 1):
                    print("your requests have been done :)")
                    break
                if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:
                    print(indices[p])


                    change_done = False
                    for subset in all_subset:
                        if change_done:
                            break
                        else:

                            if (occurred_change == change_plan["number"][i]):
                                #print("part of your request has been done :))))")
                                break

                            print("try to change, with change index {} on row {}".format(list(subset[0]),indices[p]))
                            x_train_changed[indices[p]][list(subset[0])] = 0

                            if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):

                                print(x_train[indices[p]], mnb.predict([x_train[indices[p]]])[0])
                                print(x_train_changed[indices[p]],mnb.predict([x_train_changed[indices[p]]])[0])
                                print(" \n change number {} on row {} \n".format(all_changed,indices[p]))


                                used_row.update({indices[p]: indices[p]})
                                occurred_change = occurred_change + 1
                                change_done = True
                                all_changed = all_changed + 1
                                # break

                            else:
                                x_train_changed[indices[p]] = np.copy(x_train[indices[p]])

        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        return np.copy(x_train_changed)



















































