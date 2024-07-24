import pandas as pd

from error_generator.strategies import butterfinger

#just for test
#all spams now have typo

df=pd.read_csv("../../datasets/SMSSpamCollection",sep='\t',names=['Status','Message'])
print(df.head(20))



for i in  range(df.count()[0]):
    x = df['Status'][i]
    if x=="spam":
        df.loc[i,'Message']=butterfinger(df['Message'][i])



print(df.head(20))
df.to_csv("../../datasets/SMSSpamCollection_typo")







# df['nationality']=np.where(df["first_name"]=="milad",butterfinger(df['nationality']),df['nationality'])