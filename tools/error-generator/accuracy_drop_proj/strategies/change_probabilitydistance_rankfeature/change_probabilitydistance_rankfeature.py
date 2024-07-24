import heapq
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination

"""
this method according the user request find the difference of the probability for each class and then sort the rows according to them.
for example if we have [0.1  0.6  0.3] and request of user be [0,1] we compute 0.5 for this row and sort the row Ascending
Alg4
"""

class Change_ProbabilityDistance_RankFeature(object):
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

        print("combination of feature")

        information_gain = {}
        for i in range(len(model.feature_importances_)):
            information_gain.update({i: model.feature_importances_[i]})

        ranked_information_dic = {}
        sum_gain = 0
        for L in range(0,x_train.shape[1] + 1):
            for subset in Change_Combination.combinations_index(self,information_gain.keys(), L):
                if not subset:
                    pass
                else:

                    for key in subset:
                        sum_gain = sum_gain + information_gain.get(key)
                    ranked_information_dic.update({tuple(subset): sum_gain})
                    sum_gain = 0

        print("create all subset")

        all_subset = sorted(ranked_information_dic.items(), key=lambda item: len(item[0]) * 1000 - item[1], reverse=False)




        probability = mnb.predict_proba(x_train)
        #print(probability)
        probability_distance={}

        #----------------------------------------------changing--------------------------------------------------

        for i in range(len(change_plan["key"])):
            occurred_change = 0

            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]
            #print(indices)
            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))

            probability_distance.clear()
            probability_distance_sorted=[]

            # find the distance probability between the class that user need to change

            for elements in indices:
                probability_distance.update({elements:np.abs(probability[elements][change_plan["key"][i][0]-1]- probability[elements][change_plan["key"][i][1]-1])})
            # ---------------------------finding the order of the row according to probability distance-------------------------
            # sort the row according the distance probability

            probability_distance_sorted = sorted(probability_distance.items(), key=lambda x: x[1], reverse=False)
            indices=[]
            for j in probability_distance_sorted:
                indices.append(j[0])

            #print(indices)

            print("try in indices")
            for p in range(len(indices)):

                if (all_changed == number_change_requested + 1):
                    print("your requests have been done :)")
                    break
                if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:

                    change_done = False
                    for subset in all_subset:
                        if change_done:
                            break
                        else:

                            if (occurred_change == change_plan["number"][i]):
                                #print("part of your request has been done :))))")
                                break



                            #
                            # if len(list(subset[0]))>5:
                            #     print("max number of the operations")
                            #     break

                            print("try to change, with changing index {} on row {}".format(list(subset[0]),indices[p]))

                            #######################################################
                            # impose Outlier insted of 0

                            # mean = np.mean(x_train[:,list(subset[0])])
                            # std = np.std(x_train[:,list(subset[0])])
                            # maximum = np.max(x_train[:, list(subset[0])])
                            #
                            # threshold = mean + 2 * std
                            # outlier = x_train[:,list(subset[0])][x_train[:,list(subset[0])]>threshold]
                            #
                            # if len(outlier):
                            #     x_train_changed[indices[p]][list(subset[0])] = outlier[0]
                            #
                            # else:
                            #     x_train_changed[indices[p]][list(subset[0])] = threshold +1


                            #impose of outlier the column insted of the 0
                            # x_train_changed[indices[p]][list(subset[0])] = maximum +0.1*maximum


                            #find index of values that belongs to new target
                            # indices_2 = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][1]]




                            #---------- put avg rows that belongs to new target for this specific columns
                            # print(np.mean(x_train[indices_2,list(subset[0])[0]]))
                            # x_train_changed[indices[p]][list(subset[0])] = np.mean(x_train[indices_2,list(subset[0])[0]])

                            #----------- put the first value that match to new target
                            # x_train_changed[indices[p]][list(subset[0])] = x_train_changed[indices_2[0]][list(subset[0])]


                            ########################################################
                            x_train_changed[indices[p]][list(subset[0])] = 0

                            if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):

                                print(x_train[indices[p]], mnb.predict([x_train[indices[p]]])[0])
                                print(x_train_changed[indices[p]],mnb.predict([x_train_changed[indices[p]]])[0])
                                print(" \n change number {} on row {} \n".format(all_changed, indices[p]))

                                used_row.update({indices[p]: indices[p]})
                                occurred_change = occurred_change + 1
                                change_done = True
                                all_changed = all_changed + 1
                                #break

                            else:
                                x_train_changed[indices[p]] = np.copy(x_train[indices[p]])

        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        return np.copy(x_train_changed)
