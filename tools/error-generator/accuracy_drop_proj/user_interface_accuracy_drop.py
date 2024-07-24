from accuracy_drop_proj.utilities.save_dataset.save_pickle_obj import Save_Pickle_Obj
from accuracy_drop_proj.utilities.load_dataset.load_pickle_obj import Load_Pickle_Obj
from accuracy_drop_proj.utilities.load_dataset.iris_loader import Iris_Loader
from accuracy_drop_proj.utilities.load_dataset.abalone_loader import Abalone_loader
from accuracy_drop_proj.utilities.load_dataset.digits_loader import Digits_Loader
from accuracy_drop_proj.utilities.load_dataset.eeg_loader import EEG_Loader
from accuracy_drop_proj.strategies.change_feature_one_by_one.change_feature_one_by_one import Change_Feature_One_By_One
from accuracy_drop_proj.strategies.change_feature_randomly.change_feature_randomly import Change_Feature_randomly
from accuracy_drop_proj.strategies.change_most_important_feature.change_most_important_feature import Change_Most_Important_Feature
from accuracy_drop_proj.strategies.change_combination.change_combination import Change_Combination
from accuracy_drop_proj.strategies.change_combination_min.change_combination_min import Change_Combination_Min
from accuracy_drop_proj.strategies.change_combination_feature_min.change_combination_feature_min import Change_Combination_Feature_Min
from accuracy_drop_proj.strategies.change_ranked_feature_informationgain.change_ranked_feature_informationgain import Change_Ranked_Feature_Informationgain
from accuracy_drop_proj.strategies.change_uncertaint_rankfeatures.change_uncertainty_rankfeatures import Change_Uncertainty_Rankfeatures
from accuracy_drop_proj.strategies.change_probabilitydistance_rankfeature.change_probabilitydistance_rankfeature import Change_ProbabilityDistance_RankFeature
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
from sklearn import tree
'''
Milad Abbaszadeh
Milad.abbaszadeh94@gmail.com
Bigdama Group Technical university of Berlin

'''
#---------------------- load data set -----------------------
# loader=Iris_Loader()
# loader=Abalone_loader()
# loader=Digits_Loader()
loader = EEG_Loader()
x_train, x_test, y_train, y_test = loader.load()


print(x_test.shape)

#---------------------train model -----------------------------

#----------------------MNB-------------------------------------
# print("train MNB")
# model = MultinomialNB()
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print('Accuracy of MNB classifier on test set: {:.2f}'.format(accuracy_score(y_test,y_pred)))
#----------------------SVM--------------------------------------
#
# print("train SVM")
# model = SVC(probability=True)
# model.fit(x_train, y_train)
# print('Accuracy of SVM classifier on test set: {:.2f}'.format(model.score(x_test, y_test)))

#--------------------- Tree -----------------------------------------

# model=tree.DecisionTreeClassifier(random_state= 4)
# print("train tree")
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print('Accuracy of tree classifier on test set: {:.2f}'.format(accuracy_score(y_test,y_pred)))

# ---------------------KNN ---------------------------------------
# from sklearn.naive_bayes import GaussianNB
# model= GaussianNB()
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print('Accuracy of tree classifier on test set: {:.2f}'.format(accuracy_score(y_test,y_pred)))
#-----------------------------------------------------------------------------
print("train LogisticRegression")
from sklearn.linear_model import LogisticRegression
model= LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
model.fit(x_train, y_train)
print('Accuracy of regresiion classifier on test set: {:.2f}'.format(model.score(x_test, y_test)))

#--------------------choose your method--------------------------------------------

# mymethod = Change_Feature_One_By_One() #BUG
# mymethod= Change_Feature_randomly() #BUG
# mymethod=Change_Most_Important_Feature()
# mymethod= Change_Combination()

# mymethod =Change_Combination_Min()     #Alg1
# mymethod=Change_Combination_Feature_Min() #Alg2
# mymethod = Change_Ranked_Feature_Informationgain()
# mymethod = Change_Uncertainty_Rankfeatures() #Alg3
mymethod = Change_ProbabilityDistance_RankFeature() #Alg4


#-------------------------- change plan ---------------------------------------------

start = time.time()
# change_plan={"key":[[4,9]],"number":[1]}
# change_plan={"key":[[9,7],[8,7],[10,7]],"number":[140,40,28]}



change_plan={"key":[[4,2],[3,5]],"number":[40,10]}


# change_plan={"key":[[0,1],[2,1]],"number":[10,10]}
# change_plan={"key":[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 6], [5, 7], [5, 8], [5, 9], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 7], [6, 8], [6, 9], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 8], [7, 9], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 9], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8]],"number":[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}

# change_plan={"key":[[1, 2], [1, 3], [1, 4], [1, 5], [2, 1], [2, 3], [2, 4], [2, 5], [3, 1], [3, 2], [3, 4], [3, 5], [4, 1], [4, 2], [4, 3], [4, 5], [5, 1], [5, 2], [5, 3], [5, 4]],"number":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}
out=mymethod.change(x_test,y_test,5,model,change_plan)

end = time.time()

#--------------------------- evaluation ---------------------------------------------

y_pred=model.predict(out)

print("your execuation time is {} (s)".format(end - start))
print ("\n Accuracy is: {:.2f} ".format(accuracy_score(y_test, y_pred)))
print('\n Accuracy was: {:.2f}'.format(model.score(x_test, y_test)))


#--------------------------------------------
# save the output
# saver=Save_Pickle_Obj()
# saver.save_object(out,'./outputs/results_output/output_{}_{}.pkl'.format(loader.name,start))
# # load the output
# obj_loader=Load_Pickle_Obj()
# newobj=obj_loader.load('./outputs/results_output/output_{}_{}.pkl'.format(loader.name,start))
