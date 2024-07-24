import numpy as np
#we check all combination and as far as we find the minimum combination for a selected row
#we stop there.

#check_all_combination find the possible solution for drop accuracy and doesn't care to change plan
class Change_Combination(object):
    def __init__(self):
        pass

    def combinations_index(self,iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield list(indices)

        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1

            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield list(indices)
    '''
    
    you can use this function when you would like to do Brute-force and check which label(target) can chenge to which label(target)
    then only need to write down the all possible case in the change plan.
    for this you should only provide the combination and number of change isn't important
    for example if you have 3 target you should provide 
    
    change_plan={"key":[[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]}
    
    simple code for make the change plan:
    x=[]
    for i in range(10):
        for j in range(10):
            if i!=j:
                x.append([i,j])
    print (x)
    '''

    def check_all_combination(self,x_train, y_train, percetage, mnb, change_plan):
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))


        used_row ={}
        occurred_change = 0
        all_changed = 1
        change_done = False
        x_train_changed = np.copy(x_train)

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
                    for L in range(0, len(x_train_changed[indices[p]]) + 1):
                        if change_done:

                            break
                        else:
                            for subset in self.combinations_index(x_train_changed[indices[p]], L):
                                if not subset:
                                    pass
                                else:

                                    x_train_changed[indices[p]][subset] = 0

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

    def change(self,x_train, y_train, percetage, mnb, change_plan): #check_all_combination_change_plan
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))

        used_row ={}
        occurred_change = 0
        all_changed = 1
        change_done = False
        x_train_changed = np.copy(x_train)

        for i in range(len(change_plan["key"])):
            occurred_change = 0

            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]
            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))

            for p in range(len(indices)):
                if (all_changed == number_change_requested + 1):
                    print("your requests have been done :)")
                    break
                if (occurred_change == change_plan["number"][i]):
                    print("part of your request has been done!")
                    break

                if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:

                    change_done = False
                    for L in range(0, len(x_train_changed[indices[p]]) + 1):
                        if change_done:

                            break
                        else:
                            for subset in self.combinations_index(x_train_changed[indices[p]], L):
                                if not subset:
                                    pass
                                else:

                                    x_train_changed[indices[p]][subset] = 0

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

