import pandas as pd
from sklearn.metrics import accuracy_score


###################Main Function#########################
def Bayes(py, px1y, px2y, px3y, px1, px2, px3):
    likelihood=px1y*px2y*px3y
    prior = py
    evidence=px1*px2*px3
    posterior=(likelihood*prior)/evidence
    return posterior
#########################################################



############Formating train data##########################
df_train=pd.read_csv(r"ML\Lab10\train.csv")
df_test=pd.read_csv(r"ML\Lab10\test.csv")

sexos={"male":0, "female":1}
df_train.Sex=[sexos[item] for item in df_train.Sex]
df_test.Sex=[sexos[item] for item in df_test.Sex]

df_train.Age.fillna(df_train.Age.mean(), inplace=True)
df_test.Age.fillna(df_test.Age.mean(), inplace=True)

df_train.Age=df_train.Age.astype(int)
df_test.Age=df_test.Age.astype(int)

data = [df_train, df_test]
for dataset in data:
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7
############################################################



####################Probabilities Calculation################
Class_counts=df_train['Pclass'].value_counts()
p_Class=Class_counts/len(df_train)

Sex_counts=df_train['Sex'].value_counts()
p_Sex=Sex_counts/len(df_train)

Age_counts=df_train['Age'].value_counts()
p_Age=Age_counts/len(df_train)

y_counts=df_train['Survived'].value_counts()
p_y=y_counts/len(df_train)


df_survived=df_train.loc[df_train['Survived'] == 1]
df_died=df_train.loc[df_train['Survived'] == 0]

class_survived_counts=df_survived['Pclass'].value_counts()  
p_class_survived=class_survived_counts/len(df_survived)

class_died_counts=df_died['Pclass'].value_counts()  
p_class_died=class_died_counts/len(df_died)

sex_survived_counts=df_survived['Sex'].value_counts()  
p_sex_survived=sex_survived_counts/len(df_survived)

sex_died_counts=df_died['Sex'].value_counts()  
p_sex_died=sex_died_counts/len(df_died)

age_survived_counts=df_survived['Age'].value_counts()  
p_age_survived=age_survived_counts/len(df_survived)

age_died_counts=df_died['Age'].value_counts()  
p_age_died=age_died_counts/len(df_died)
##############################################################



###########################Making Preidction##############################
result_array=[]

for i in range(0,418):
    feature_class=df_test.iloc[i]['Pclass']
    feature_sex=df_test.iloc[i]['Sex']
    feature_age=df_test.iloc[i]['Age']
    
    P_Y1=Bayes(p_y[1], p_class_survived[feature_class], p_sex_survived[feature_sex], p_age_survived[feature_age], p_Class[feature_class], p_Sex[feature_sex], p_Age[feature_age])
    P_Y0=Bayes(p_y[0], p_class_died[feature_class], p_sex_died[feature_sex], p_age_died[feature_age], p_Class[feature_class], p_Sex[feature_sex], p_Age[feature_age])
    
    if P_Y0 > P_Y1:
        result=0
    else:
        result=1
        
    result_array.append(result)
##############################################################################



#################################Printing Output#########################################
actual = pd.read_csv(r"ML\Lab10\gender_submission.csv",usecols=["Survived"])
actual.head()
actual_array = actual.Survived.to_list()


testPassenger = pd.read_csv(r"ML\Lab10\gender_submission.csv",usecols=["PassengerId"])
testPassenger.head()
testPassenger_array = testPassenger.PassengerId.to_list()

print('length of test set= ',len(actual_array))

for p_id,act, pred in zip(testPassenger_array,actual_array, result_array):
    print('Passenger Id = ',p_id,' : Actual Result = ',act, '  Predicted result = ',pred)


print("Accuracy: ", accuracy_score(actual_array, result_array)*100, " %", sep='')
###########################################################################################




























###FORMULAES or CONCEPTS USED :-

#particular feature in a class is unrelated to the presence of any other feature.(knows ass naive)

#Recommendation System -- NB + data mining

#Text classification/ Spam Filtering -- they require small number of training data for estimating the parameters necessary for classification.

#Zero Frequency -- smoothing technique -- Laplace estimation.

# MAP(h) = max(P(h|d))
# or
# MAP(h) = max((P(d|h) * P(h)) / P(d))
# or
# MAP(h) = max(P(d|h) * P(h))
# MAP(h) = max(P(d|h))        if h is uniform, h is even.

#Nevertheless, the approach performs surprisingly well on data where this assumption does not hold.

#For continuous distributions, the Gaussian naive Bayes is the algorithm of choice. For discrete features, multinomial and Bernoulli distributions as popular.

#Training is fast because only the probability of each class and the probability of each class given different input (x) values need to be calculat