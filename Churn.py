#Author: Nitish Arora
#Purpose: build a simple Churn model using Kaggle data from here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/versions/1?resource=download
#This data is then saved locally
#Credit to Natassha Selvaraj and 365 data science for the inspiration:
#https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/

#Import relevant packages. Make sure these are loaded and installed in the environment e.g. pip, conda etc.
import pandas as pd #All basic stuff
import numpy as np #Numbers
import matplotlib.pyplot as plt #charts
import seaborn as sns  #for diff plot types
import sklearn #for ML modelling things. I use Pythn 3.8, had to install sklearn = 1.1 for it to work
##import imblearn

#Define where to read data from, Note: Data file renamed to "Data.csv"
path = r'C:\Users\nitis\Documents\GitHub\Funn_Scripts\Python\Data Analysis\Churn\Data.csv'
df = pd.read_csv(path)

#Basic data exploration and findings on the data
##print(df.head())
##df.info()
##print(df["Churn"].value_counts())

#FINDINGS:
#Each user is identified through a unique customer ID.
#There are 19 independent variables used to predict the target feature – customer churn.
#In this dataset, customer churn is defined as users who have left within the last month.
#Out of 7042 customers, 27% of the customers (1869) churned, 73% (5174) did not

#Step 1: Exploring the data, simple analysis, visualisations, etc.
#Technically optional, but this stuff really helps yuo understand the data better

#Analysis 1: Understanding the data and key columns of interest
#Findings: Gender has equal split, More young people than seniors, rouhgly equal with partnerds ands single, v few with dependents
cols = ['gender','SeniorCitizen',"Partner","Dependents"]
numerical = cols

plt.figure(figsize=(20,4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=df)
    ax.set_title(f"{col}")

#Print the plot
##plt.show()

#Analysis 2: Understanding the relationship between churn and monthly charges
#Findings: Shows that monthly charges has an effect on churn (median churning customers paid higher)

sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

##plt.show()

#Analysis 3: Understanding the relationship between churn and key columns of interest
#Findings: Shows what type of internet service (fiber optic), techsupport (no techsupport is more churn)
# onine backup (those who dont prefer churn more), contract type (month-on-month churn more
cols = ['InternetService',"TechSupport","OnlineBackup","Contract"]

plt.figure(figsize=(14,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x ="Churn", hue = str(col), data = df)
    ax.set_title(f"{col}")

##plt.show()

#Step 2: Data cleaining
#1. Converting all numerical factors to numeric. The one to change is Total Charges

df['TotalCharges'] = df['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

#2. Turning / encoding categorical factors to numeric, step-wise. Currently these are charcetrs (e.g. Yes/No etc)

from sklearn import preprocessing

#Allows all categorical variables to be encoded
cat_features = df.drop(['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure'],axis=1)
le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()

#Creates finaldf with joined
num_features = df[['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)

#Step 3: Do a train/test split

from sklearn.model_selection import train_test_split

finaldf = finaldf.dropna()
finaldf = finaldf.drop(['customerID'],axis=1)

X = finaldf.drop(['Churn'],axis=1)
y = finaldf['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #test_size set to 33% but can be moved around

#Step 4: Oversampling. This is optional. Ideally want a balanced set between both churning and non-churning customers, but 27% is pretty good imo

##from imblearn.over_sampling import SMOTE
##
##oversample = SMOTE(k_neighbors=5)
##X_smote, y_smote = oversample.fit_resample(X_train, y_train)
##X_train, y_train = X_smote, y_smote
##y_train.value_counts(): #Should yield equal for both customer types

#Step 5: Build the model and test the outcome. TO GIVE MORE OPTIONS HERE!!!!

#Build the model using train data
#2 recommended models, Random Forest and GBMs, suggest we use the latter
#Below, GBM outperforms the RF and it can perform better if we tune it appropriately

#RANDOM FOREST - currently commented out
#Tuning the RF is simple: number of trees is set trees as 100. We test this and select a good number with high accuracy
#Everything else equal, lowest trees so that its computationally better
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(random_state=46,n_estimators = 100)

#GBM - default mdoel
#Tuning the GBM is harder: many hyperparameters to test and tune, but GBMs tend to be more accurate than RFs
#Everything else equal, go with the best computational model
#To know more about tuning, look at this link= https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0) #Currently set to default values

#Fit the model
model.fit(X_train,y_train)

#Test results using test data
from sklearn.metrics import accuracy_score

preds = model.predict(X_test)
print("The accuracy score of the model is:")
print(accuracy_score(preds,y_test))

#Look at important features of the model
importances = model.feature_importances_
#
# Sort the feature importance in descending order
#
sorted_indices = np.argsort(importances)[::-1]
 
feat_labels = df.columns[1:]

print("The important features are (sorted by most to least importance)")
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[sorted_indices[f]],
                            importances[sorted_indices[f]]))
