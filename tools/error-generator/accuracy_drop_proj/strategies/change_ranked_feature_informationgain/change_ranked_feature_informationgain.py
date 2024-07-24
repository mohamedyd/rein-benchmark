from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination
import numpy as np
class Change_Ranked_Feature_Informationgain(object):
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

        #find the order of the feature according to information gain
        model = ExtraTreesClassifier()
        model.fit(x_train, y_train)


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

        all_subset = sorted(ranked_information_dic.items(), key=lambda item: len(item[0]) * 1000 - item[1], reverse=False)



        #changing
        for i in range(len(change_plan["key"])):
            occurred_change = 0

            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]
            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))

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
                                #                         print("part of your request has been done :))))")
                                break


                            print("try to change with change index {}".format(list(subset[0])))
                            x_train_changed[indices[p]][list(subset[0])] = 0

                            if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):

                                print(x_train[indices[p]], mnb.predict([x_train[indices[p]]])[0])
                                print(x_train_changed[indices[p]],
                                      mnb.predict([x_train_changed[indices[p]]])[0])
                                print(" \n change number {} \n".format(all_changed))

                                used_row.update({indices[p]: indices[p]})
                                occurred_change = occurred_change + 1
                                change_done = True
                                all_changed = all_changed + 1
                                break

                            else:
                                x_train_changed[indices[p]] = np.copy(x_train[indices[p]])




        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        return np.copy(x_train_changed)

