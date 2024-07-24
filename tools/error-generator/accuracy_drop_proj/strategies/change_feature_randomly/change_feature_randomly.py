import numpy as np
import random

#try to change the target by changing the features randomly

class Change_Feature_randomly(object):
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan):
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))
        used_row = {}
        col_history = []
        occurred_change = 0
        all_changed = 1
        x_train_changed = np.copy(x_train)

        for i in range(len(change_plan["number"])):
            occurred_change = 0
            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]

            for p in range(len(indices)):
                if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:

                    while (len(col_history) <= x_train.shape[1]):  # range 4
                        col = random.randint(0, x_train.shape[1] - 1)
                        while col in col_history:
                            col = random.randint(0, x_train.shape[1] - 1)
                            if (len(col_history) == x_train.shape[1]):
                                break
                        col_history.append(col)

                        if occurred_change == change_plan["number"][i]:
                            col_history = []
                            break

                        x_train_changed[indices[p]][col] = 0

                        if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):

                            print(x_train[indices[p]], mnb.predict([x_train[indices[p]]])[0])
                            print(x_train_changed[indices[p]], mnb.predict([x_train_changed[indices[p]]])[0])
                            print(" \n change number {} \n".format(all_changed))
                            used_row.update({indices[p]: indices[p]})
                            occurred_change = occurred_change + 1
                            all_changed = all_changed + 1
                            col_history = []
                            break

                        else:
                            x_train_changed[indices[p]] = np.copy(x_train[indices[p]])

        if (all_changed < number_change_requested - 1):
            print("your request doesn't complete! please change your plan")
        return np.copy(x_train_changed)
