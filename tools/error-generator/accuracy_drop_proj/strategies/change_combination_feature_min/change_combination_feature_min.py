import numpy as np
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination
# for each combination (from len1 to n) check the target changes
#e.g first check all row that can change them only with changing one feature then two feature and so on
#algorithm 2
class Change_Combination_Feature_Min(object):
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan):#check_combination_change_plan_features
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))


        used_row ={}
        occurred_change = 0
        all_changed = 1

        x_train_changed = np.copy(x_train)

        for i in range(len(change_plan["key"])):

            occurred_change = 0
            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]


            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))

            for L in range(0, len(x_train_changed[0]) + 1 ):
                print("changing target, with change {} features ----".format(L))

                for subset in Change_Combination.combinations_index(self,x_train_changed[0], L):
                    if not subset:
                        pass
                    else:
                        if (occurred_change == change_plan["number"][i]):
                            #print("part of your request has been done :))))")
                            break


                        for p in indices:


                            if y_train[p] == mnb.predict([x_train[p]]) and p not in used_row:
                                change_done = False

                                if change_done:
                                    break
                                else:

                                    if (occurred_change == change_plan["number"][i]):

                                        break
                                    else:

                                        x_train_changed[p][subset] = 0

                                        if (change_plan["key"][i][1] == mnb.predict([x_train_changed[p]])[0]):

                                            change_done = True
                                            print("with change features index number {} row number {} has been changed".format(subset,p))
                                            print(x_train[p], mnb.predict([x_train[p]])[0])
                                            print(x_train_changed[p], mnb.predict([x_train_changed[p]])[0])

                                            print(" \n change number {} \n".format(all_changed))
                                            used_row.update({p:p})
                                            occurred_change = occurred_change + 1
                                            all_changed = all_changed + 1

                                        else:
                                            x_train_changed[p] = np.copy(x_train[p])


        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        else:
            print("your request is done :)")


        return np.copy(x_train_changed)

