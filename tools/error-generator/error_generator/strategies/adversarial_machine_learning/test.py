
# coding: utf-8

# # load the data and first test the base line 

# In[1]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
import numpy as np


# In[2]:


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

# print the iris data
print(iris.data)


# print the iris data
# print(iris.data)


# print the names of the four features
print(iris.feature_names)


# print integers representing the species of each observation# print 
print(iris.target)


# In[3]:


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# In[4]:


# store feature matrix in "X"# store  
X = iris.data

# store response vector in "y"
y = iris.target


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[6]:


mnb = MultinomialNB()
mnb.fit(x_train,y_train)


# # result of the model  on test data (First step)

# In[7]:


#compute the f1 and accuracy

y_pred=mnb.predict(x_test)

print ("f1 on test data is       {}".format(f1_score(y_test, y_pred, average='macro')))
print ("Accuracy on test data is {}".format(accuracy_score(y_test, y_pred)))


# In[8]:


#3 change for 10 percentage less accuracy
print(y_test[:10])
print(y_pred[:10])
y_pred[:10]= [0,1,1,0,1,1,1,0,0,1]

print ("Accuracy on test data is {}".format(accuracy_score(y_test, y_pred)))


# # result model on the same train data

# In[11]:


y_pred_train=mnb.predict(x_train)


print ("f1 on same train data is       {}".format(f1_score(y_train, y_pred_train, average='macro')))
print ("Accuracy on same train data is {}".format(accuracy_score(y_train, y_pred_train)))
print("\n" + classification_report(y_train, y_pred_train))

print(x_train.shape)
print(confusion_matrix(y_train, y_pred_train))


# # base line
# - change one feature and chek the target (change function)
# - change randomly one value and check the target(change random)

# In[35]:


def change(x_train,y_train,percetage,mnb):
    
    number_change = int(percetage/100*x_train.shape[0])
    print(number_change)
    
    x_train_changed = np.copy(x_train)

    for count,ele in enumerate(x_train_changed[:number_change]):
        print(x_train[count],y_train[count])
        for i in range(4):
            x_train_changed[count][i] = 0

            if (y_train[count] != mnb.predict([x_train_changed[count]])):
                break
            else:
                x_train_changed[count]= np.copy(x_train[count])

        print(x_train_changed[count],mnb.predict([x_train_changed[count]])[0])
        print(" \n change number {} \n".format(count))

    return np.copy(x_train_changed)


# In[36]:


# new=change(x_train,y_train,20.8,mnb)
new=change(x_test,y_test,20,mnb)


# # result model on the dirty data (10 %)

# In[37]:


y_pred_train_changed=mnb.predict(new)


# print ("f1 on same train data is       {}".format(f1_score(y_train, y_pred_train_changed, average='macro')))
# print ("Accuracy on same train data is {}".format(accuracy_score(y_train, y_pred_train_changed)))


print ("f1 on same train data is       {}".format(f1_score(y_test, y_pred_train_changed, average='macro')))
print ("Accuracy on test data is {}".format(accuracy_score(y_test, y_pred_train_changed)))


# In[16]:


def change_random(x_train,y_train,percetage,mnb):
    number_change_requested = int(percetage/100*x_train.shape[0])
    
    print("{} percentage error is equal to {} change \n".format(percetage,number_change_requested))
    row_history=[]
    col_history=[]
    
    x_train_changed_random = np.copy(x_train)
  
    for i in range(number_change_requested):
              
        row = random.randint(1, x_train.shape[0])
        col = random.randint(0,x_train.shape[1]-1)

        while (row in row_history) or (col in col_history):
            
            row = random.randint(1, x_train.shape[0])
            col = random.randint(0,x_train.shape[1]-1)
        row_history.append(row)
        col_history.append(col)


        print("row {}   value {} {}".format(row,x_train[row],y_train[row]))
        
        while(y_train[row] == mnb.predict([x_train_changed_random[row]])):
            x_train_changed_random[row] = np.copy(x_train[row])
            x_train_changed_random[row][col]=0
            col = random.randint(0,x_train.shape[1]-1)
                
            
        print("row {}   value {} {}".format(row,x_train_changed_random[row],mnb.predict([x_train_changed_random[row]])[0]))
#         print('"""""""""""""""""')
        print(" \n change number {} \n".format(i))
        col_history=[]
    return np.copy(x_train_changed_random)
         



# In[21]:


new_random=change_random(x_test,y_test,20,mnb)


# # result model on the dirty data (10 %) -random

# In[22]:


y_pred_test_changed_random=mnb.predict(new_random)


# print ("f1 on same train data is       {}".format(f1_score(y_train, y_pred_train_changed_random, average='macro')))
# print ("Accuracy on same train data is {}".format(accuracy_score(y_train, y_pred_train_changed_random)))


print ("f1 on same train data is       {}".format(f1_score(y_test, y_pred_train_changed_random, average='macro')))
print ("Accuracy on same train data is {}".format(accuracy_score(y_test, y_pred_train_changed_random)))


# # find the most important features

# In[38]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier


# In[39]:


#knn = KNeighborsClassifier(n_neighbors=4)
mnb2 = MultinomialNB()
# sfs1 = SFS(knn, 
#            k_features=1, 
#            forward=True, 
#            floating=False, 
#            verbose=2,
#            scoring='accuracy',
#            cv=0)
sfs1 = SFS(mnb, 
#            k_features=1, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)






sfs1 = sfs1.fit(X, y)


# In[40]:


sfs1.subsets_


# # most important feature

# In[41]:


sfs1.k_feature_idx_


# In[43]:


def change_spetial_feature(x_train,y_train,percetage,mnb,feature):
    change_item=0
    number_change_spetial = int(percetage/100*x_train.shape[0])
    print(number_change_spetial)
    
    x_train_changed_spetial = np.copy(x_train)

    for count,ele in enumerate(x_train_changed_spetial):
        

        x_train_changed_spetial[count][feature] = 0

        if (y_train[count] != mnb.predict([x_train_changed_spetial[count]])):
            change_item=change_item+1
            if (change_item < number_change_spetial):

                print(x_train[count],y_train[count])
                print(x_train_changed_spetial[count],mnb.predict([x_train_changed_spetial[count]])[0])
                print(" \n change number {} \n".format(change_item))
            else:

                break
        else:
            x_train_changed_spetial[count]= np.copy(x_train[count])

    return np.copy(x_train_changed_spetial)


# In[46]:


new_random_spetial=change_spetial_feature(x_test,y_test,20,mnb,0)


# # result model on the dirty data (10 %) -spetial feature

# In[47]:


y_pred_train_changed_spetial=mnb.predict(new_random_spetial)


# print ("f1 on same train data is       {}".format(f1_score(y_train, y_pred_train_changed_spetial, average='macro')))
# print ("Accuracy on same train data is {}".format(accuracy_score(y_train, y_pred_train_changed_spetial)))

print ("f1 on same train data is       {}".format(f1_score(y_test, y_pred_train_changed_spetial, average='macro')))
print ("Accuracy on same train data is {}".format(accuracy_score(y_test, y_pred_train_changed_spetial)))


# In[48]:


co=0
for count1,ele1 in enumerate(new_random_spetial):
    if ele1[0]==0:
        co=co+1
        print(new_random_spetial[count1])
print(co)

