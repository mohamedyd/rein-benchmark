import random
import numpy as np

#change the features one by one and start from feature 0 and as far as the target change
# we stop there
class Change_Feature_One_By_One(object):
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan):
        number_change = int(percetage / 100 * x_train.shape[0])

        x_train_changed = np.copy(x_train)
        row_history = {}
        try_times = 0
        occurred_change = 0
        change_done = False

        for i in range(len(change_plan["number"])):
            for j in range(change_plan["number"][i]):

                change_done = False
                while (change_done == False):

                    # find uniqe row & as same as user request
                    row = random.randint(1, x_train.shape[0] - 1)
                    try_times = 0
                    while (mnb.predict([x_train[row]])[0] != change_plan["key"][i][0] or row in row_history):  # row in row_history or

                        row = random.randint(1, x_train.shape[0] - 1)
                        if try_times > len(y_train):
                            print("your request is not possiable")
                            break

                        try_times = try_times + 1
                    if try_times > len(y_train):
                        print("part of your request can't run")
                        break

                    row_history.update({row:row})

                    for ii in range(x_train.shape[1]):  # range(4)
                        x_train_changed[row][ii] = 0

                        if (change_plan["key"][i][1] == mnb.predict([x_train_changed[row]])):
                            print(x_train[row], mnb.predict([x_train[row]])[0])
                            print(x_train_changed[row], mnb.predict([x_train_changed[row]])[0])
                            occurred_change = occurred_change + 1
                            change_done = True
                            print(" \n change number {} \n".format(occurred_change))
                            break
                        else:
                            #                         print("this change doesnot run")
                            x_train_changed[row] = np.copy(x_train[row])

        return np.copy(x_train_changed)