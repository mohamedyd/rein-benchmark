import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination
from sklearn.neighbors import KNeighborsClassifier


'''
This function first find the index of the feature that have a maximum information gain then only try to change that feature and then go to
combination with lenth 2 and try to change that two feature and so on  
'''

class Change_Most_Important_Feature(object):
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan):
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))

        #find the most important feature

        sfs = SFS(mnb,
                   k_features=len(x_train[0]),
                   forward=True,
                   floating=False,
                   verbose=2,
                   scoring='accuracy',
                   cv=5)
        pipe = make_pipeline(StandardScaler(), sfs)
        pipe.fit(x_train, y_train)

        #-------------plotting------------------
        fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.show()

        #get future of the sfs order and only change them.
        x_train_changed = np.copy(x_train)
        used_row ={}
        all_changed = 1

        for i in range(len(change_plan["key"])):

            occurred_change = 0
            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]

            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))

            for L in range(1, len(sfs.subsets_) + 1): #number of the features
                subset=list(sfs.subsets_[L]['feature_idx'])

                if (occurred_change == change_plan["number"][i]):
                    break
                print("change feature index {} ----".format(subset))
                for p in range(len(indices)):
                    x_train_changed[indices[p]][subset] = 0

                    if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:

                        if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):

                            print("with change features index {} row number {} has been changed".format(subset,
                                                                                                         indices[
                                                                                                             p]))
                            print(x_train[indices[p]], mnb.predict([x_train[indices[p]]])[0])
                            print(x_train_changed[indices[p]], mnb.predict([x_train_changed[indices[p]]])[0])

                            print(" \n change number {} \n".format(all_changed))
                            used_row.update({indices[p]:indices[p]})
                            occurred_change = occurred_change + 1
                            all_changed = all_changed + 1

                            if (occurred_change == change_plan["number"][i]):
                                print("part of your request has been done :)")
                                break
                        else:
                            x_train_changed[indices[p]] = np.copy(x_train[indices[p]])
                    else:
                        x_train_changed[indices[p]] = np.copy(x_train[indices[p]])

            #check for rest of the possible changes


                    # for LL in range(0, len(x_train_changed[0]) + 1):
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

                for subsets in Change_Combination.combinations_index(self,x_train_changed[0], L):
                    if (subset != subsets):
                        if not subsets:
                            pass
                        else:
                            if (occurred_change == change_plan["number"][i]):
                                #print("part of your request has been done :))))")
                                break
                            print("change feature index {} ----".format(subsets))
                            for pp in range(len(indices)):
                                x_train_changed[indices[pp]][subsets] = 0

                                if y_train[indices[pp]] == mnb.predict([x_train[indices[pp]]]) and indices[pp] not in used_row:

                                    if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[pp]]])[0]):

                                        print("with change features index {} row number {} has been changed".format(subsets,indices[pp]))
                                        print(x_train[indices[pp]], mnb.predict([x_train[indices[pp]]])[0])
                                        print(x_train_changed[indices[pp]], mnb.predict([x_train_changed[indices[pp]]])[0])

                                        print(" \n change number {} \n".format(all_changed))
                                        used_row.update({indices[pp]:indices[pp]})
                                        occurred_change = occurred_change + 1
                                        all_changed = all_changed + 1

                                        if (occurred_change == change_plan["number"][i]):
                                            print("part of your request has been done :)")
                                            break
                                    else:
                                        x_train_changed[indices[pp]] = np.copy(x_train[indices[pp]])
                                else:
                                    x_train_changed[indices[pp]] = np.copy(x_train[indices[pp]])
                    else:
                        print("subsets are equal {}----------------------------------------------".format(subsets))




        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        else:
            print("your request is done :)")

        return np.copy(x_train_changed)

