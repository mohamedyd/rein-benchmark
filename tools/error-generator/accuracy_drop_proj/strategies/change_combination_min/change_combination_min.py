import numpy as np
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination
import matplotlib.pyplot as plt; plt.rcdefaults()


#check all possible combination and then change the minimum one
# e.g first rows that require only one change for changing the target and then rows by two feature and so on
#algorithm 1
class Change_Combination_Min(object):#base rows
    def __init__(self):
        pass

    def change(self,x_train, y_train, percetage, mnb, change_plan): #check_combination_change_plan_probability
        number_change_requested = int(percetage / 100 * x_train.shape[0])
        print("{} percentage error is equal to {} change \n".format(percetage, number_change_requested))


        used_row={}
        occurred_change = 0
        possible_changes_counter = 0
        all_changed = 1
        change_done = False
        x_train_changed = np.copy(x_train)
        possible_changes = {}  # key: number of changes and  value:[row,[columns should change]]

        for i in range(len(change_plan["key"])):
            possible_changes_counter = 0
            occurred_change = 0
            indices = [t for t, x in enumerate(y_train) if x == change_plan["key"][i][0]]
            possible_changes = {x: [] for x in range(len(x_train[0]) + 1)}
            print("{} rows have target {} \n".format(len(indices), change_plan["key"][i][0]))
            print(indices)
            for p in range(len(indices)):

                if y_train[indices[p]] == mnb.predict([x_train[indices[p]]]) and indices[p] not in used_row:
                    change_done = False

                    for L in range(0, len(x_train_changed[indices[p]]) + 1 ):
                        if change_done:
                            break
                        else:
                            for subset in Change_Combination.combinations_index(self,x_train_changed[indices[p]], L):
                                if not subset:
                                    pass
                                else:
                                    #print('subset is {} and row is {}'.format(subset,indices[p]))
                                    x_train_changed[indices[p]][subset] = 0
                                    if (possible_changes_counter == change_plan["number"][i]):
                                        # print("the requested number of change have been found")
                                        break
                                    else:

                                        if (change_plan["key"][i][1] == mnb.predict([x_train_changed[indices[p]]])[0]):
                                            possible_changes[len(subset)].append([indices[p], subset])
                                            change_done = True
                                            x_train_changed[indices[p]] = np.copy(x_train[indices[p]])
                                            possible_changes_counter=possible_changes_counter+1
                                            #print('Found')
                                            break
                                        else:
                                            x_train_changed[indices[p]] = np.copy(x_train[indices[p]])

            if (all(value == [] for value in possible_changes.values())):
                print("part of your request not possible!")
                break

            for key in sorted(possible_changes):
                if (occurred_change == change_plan["number"][i]):
                    break
                print("there are {} candidate for changing target with change {} features".format(len(possible_changes[key]), key))
                variable = possible_changes[key]
                for t in range(len(variable)):

                    print(x_train[variable[t][0]], mnb.predict([x_train[variable[t][0]]])[0])

                    x_train_changed[variable[t][0]][variable[t][1]] = 0
                    print(x_train_changed[variable[t][0]], mnb.predict([x_train_changed[variable[t][0]]])[0])
                    print(" \n change number {} on row {} \n".format(all_changed,variable[t][0]))
                    used_row.update({variable[t][0]:variable[t][0]})
                    occurred_change = occurred_change + 1
                    all_changed = all_changed + 1
                    if (occurred_change == change_plan["number"][i]):
                        print("part of your request has been done :)")
                        break

            #plotting

            print("----plotting----")
            x_pos = (range(0, len(x_train_changed[indices[p]]) + 1))
            y_pos = np.arange(len(x_train_changed[indices[p]]) + 1)
            chart_freq=[]
            print("number of feature,how many changes is possible")
            for key, value in possible_changes.items():
                print(key, len([item for item in value if item]))
                chart_freq.append(len([item for item in value if item]))

            fig = plt.figure()
            outputFile = "./outputs/fig_output/change_combination_min/request{}.png".format(i)
            plt.bar(y_pos, chart_freq, align='center', alpha=0.5)
            plt.xticks(y_pos, x_pos)
            plt.ylabel('frequency')
            plt.xlabel('with changing X feature you can change target')
            plt.title('Summary of your request for change target {}'.format(change_plan['key'][i]))
            fig.savefig(outputFile)



        if (all_changed <= number_change_requested):
            print("your request doesn't complete! please change your plan")
        else:
            print("your request is done :)")


        return np.copy(x_train_changed)

