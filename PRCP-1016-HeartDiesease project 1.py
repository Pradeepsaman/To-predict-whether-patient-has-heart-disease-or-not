#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Business case:To predict whether patient has heart disease or not
# as its a Binary Classification

# In[19]:


#import the data
data=pd.read_csv('valuesmerge.csv') #reading the dataset
data


# In[20]:


pd.set_option('display.max_rows',None) # to display all the rows 


# In[21]:


data


# In[22]:


pd.reset_option('display.max_rows',None)


# In[23]:


data


# # Domain Analysis
# •	There are 14 columns in the dataset, where the patient_id column is a unique and random identifier. The remaining 13 features are described in the section below.
# 
# •	heart_disease_present: 0-No, 1-yes
# 
# •	slope_of_peak_exercise_st_segment (type: int): the slope of the peak exercise ST segment, an electrocardiography read out indicating quality of blood flow to the heart
# 
# •	thal (type: categorical): results of thallium stress test measuring blood flow to the heart, with possible values normal, fixed_defect, reversible_defect
# 
# •	resting_blood_pressure (type: int): resting blood pressure
# 
# •	chest_pain_type (type: int): chest pain type (4 values)
# 
# •	num_major_vessels (type: int): number of major vessels (0-3) colored by flourosopy
# 
# •	fasting_blood_sugar_gt_120_mg_per_dl (type: binary): fasting blood sugar > 120 mg/dl
# 
# •	resting_ekg_results (type: int): resting electrocardiographic results (values 0,1,2)
# 
# •	serum_cholesterol_mg_per_dl (type: int): serum cholestoral in mg/dl
# 
# •	oldpeak_eq_st_depression (type: float): oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms
# 
# •	sex (type: binary): 0: female, 1: male
# 
# •	age (type: int): age in years
# 
# •	max_heart_rate_achieved (type: int): maximum heart rate achieved (beats per minute)
# 
# •	exercise_induced_angina (type: binary): exercise-induced chest pain (0: False, 1: True)

# # Basic Checks

# In[24]:


data.head() # to view first five rows


# In[25]:


data.tail() # to view last five rows


# In[26]:


data.info #to check datatype and non,null values of all columns


# In[27]:


data.describe() #used to view some statistical information such as mean,std,count,min,max,etc..


# In[28]:


data.describe(include='O')# to display info of categorical data


# ### Get all the rows where num_major_vessels= 0

# In[29]:


len(data.loc[data['num_major_vessels']==0]) 


# ### Get all the rows where fasting_blood_sugar_gt_120_mg_per_dl= 0

# In[30]:


len(data.loc[data['fasting_blood_sugar_gt_120_mg_per_dl']==0])


# ### Get all the rows where exercise_induced_angina= 0

# In[31]:


len(data.loc[data['exercise_induced_angina']==0])


# In[32]:


###  hence num_major_vessels ,fasting_blood_sugar_gt_120_mg_per_dl and exercise_induced_angina are containing zeros more than 50% of the rows so dropping all these three columns

#dropping thal as its categorical feature

data.drop(['num_major_vessels','exercise_induced_angina','thal','fasting_blood_sugar_gt_120_mg_per_dl'],axis=1,inplace=True)


# In[33]:


data


# In[34]:


data.drop(index=180, inplace=True, axis=0)


# ### Get the missing values

# In[35]:


data.isnull().sum()


# ### Hence there are no missing values in our data

# # Exploratory Data Analysis(EDA)

# ## Univariate Analysis

# In[36]:


get_ipython().system('pip install sweetviz')


# In[41]:


import sweetviz as sv #importing sweetviz library
my_report=sv.analyze(data) # syntax to use sweetviz
my_report.show_html() #default arguments will generate to sweetviz_report.html


# ### Report SWEETVIZ_REPORT.html was generated!

# ### Insights from Univartiate Analysis
# Majority of patients are Male
# 
# 65% of patients are Male
# 
# 25% of patients are Female
# 
# Majority of patients are in the age from 55 to 65years old
# 
# patients who do exercise are on less chance of getting chest pain
# 
# Maximum heartrate acheieved is 170 beates per minute
# 
# Most of the people have Cholesterol ranges from 220 to 280 mg per dl
# 
# Highest Cholesterol is 250 mg per dl
# 
# patients having 4values are high on getting chest pain
# 
# patients having 1values are less chance of getting chest pain

# ###  Bivariate Analysis

# In[42]:


data.columns


# In[43]:


data1=data[['heart_disease_present',
       'slope_of_peak_exercise_st_segment', 'resting_blood_pressure',
       'chest_pain_type', 'resting_ekg_results', 'serum_cholesterol_mg_per_dl',
       'oldpeak_eq_st_depression', 'sex', 'age', 'max_heart_rate_achieved']]


# In[44]:


len(data.loc[data['slope_of_peak_exercise_st_segment']==0]) 


# In[45]:


len(data.loc[data['resting_blood_pressure']==0]) 


# In[46]:


len(data.loc[data['chest_pain_type']==0]) 


# In[47]:


len(data.loc[data['serum_cholesterol_mg_per_dl']==0]) 


# In[48]:


len(data.loc[data['serum_cholesterol_mg_per_dl']==0]) 


# In[49]:


len(data.loc[data['oldpeak_eq_st_depression']==0]) 


# In[50]:


len(data.loc[data['max_heart_rate_achieved']==0]) 


# In[51]:


data1


# ## Let'see how data is distributed in every column

# In[52]:


plt.figure(figsize=(20,25), facecolor='white')#defining  canvas size
plotnumber = 1 #maintian count for graph

for column in data1:#accessing the columns
    if plotnumber<=10 :# as there are 10 columns in the data
        ax = plt.subplot(5,5,plotnumber)# plotting 10 graphs (5-rows,5-columns) ,plotnumber is for count 
        sns.distplot(data1[column])#plotting dist plot to know distribution
        plt.xlabel(column,fontsize=10)
    plotnumber+=1
plt.show()


# In[53]:


data1.columns


# ### Note:
# Normal distribution curce -heart_disease_present, slope_of_peak_exercise_st_segment , resting_blood_pressure , resting_ekg_results, serum_cholesterol_mg_per_dl ,sex , age, max_heart_rate_achieved
# 
# Not Normal distribution- chest_pain_type , oldpeak_eq_st_depression

# # Checking Outliers

# In[54]:


plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1
for column in data1:
    if plotnumber<=10:
        ax=plt.subplot(5,5,plotnumber)
        sns.boxplot(x=data1[column],hue=data1.heart_disease_present)
        plt.xlabel(column,fontsize=10)
        plt.ylabel('heart_disease_present',fontsize=20)
    plotnumber+=1
plt.show()
plt.tight_layout()


# In[55]:


data1['oldpeak_eq_st_depression']=data1['oldpeak_eq_st_depression'].replace(0,data1['oldpeak_eq_st_depression'].median())


# In[56]:


len(data1.loc[data1['oldpeak_eq_st_depression']==0]) 


# ### Hence all non,null zero values are handled

# In[57]:


#to manage outliers

from scipy import stats


# In[58]:


sns.boxplot(data1.resting_blood_pressure)


# In[59]:


data1


# In[60]:


IQR=stats.iqr(data1.resting_blood_pressure, interpolation='midpoint')
IQR
Q1=data1.resting_blood_pressure.quantile(.25)
Q3=data1.resting_blood_pressure.quantile(.75)
maxm=Q3+IQR*1.5
minm=Q1-IQR*1.5


# In[61]:


IQR


# In[62]:


maxm


# In[63]:


minm


# In[64]:


len(data1.loc[data1['resting_blood_pressure']>maxm, 'resting_blood_pressure'])#number of columns that contain outliers


# In[65]:


percentage_of_outliers=6*100/180
print(percentage_of_outliers)


# In[66]:


data1.loc[data1['resting_blood_pressure']<minm, 'resting_blood_pressure']


# In[67]:


len(data1.loc[data1['resting_blood_pressure']<minm, 'resting_blood_pressure'])


# In[68]:


np.median(data1.resting_blood_pressure)


# ###  Replacing Outliers with Median

# In[69]:


data1.loc[data1['resting_blood_pressure']>maxm, 'resting_blood_pressure']=np.median(data1.resting_blood_pressure)


# In[70]:


data1.loc[data1['resting_blood_pressure']<minm, 'resting_blood_pressure']=np.median(data1.resting_blood_pressure)


# In[71]:


data1.loc[data1['resting_blood_pressure']>maxm, 'resting_blood_pressure'] #removed outliers from the column 


# In[72]:


#plotting graph after removing outliers

sns.boxplot(data1.resting_blood_pressure)


# ### Hence outliers of resting_blood_pressure has been handled

# In[73]:


sns.boxplot(data1.chest_pain_type)


# In[74]:


IQR=stats.iqr(data1.chest_pain_type, interpolation='midpoint')
IQR
Q1=data1.chest_pain_type.quantile(.25)
Q3=data1.chest_pain_type.quantile(.75)
maxm=Q3+IQR*1.5
minm=Q1-IQR*1.5


# In[75]:


len(data1.loc[data1.chest_pain_type>maxm, 'chest_pain_type'])


# In[76]:


len(data1.loc[data1.chest_pain_type<minm, 'chest_pain_type'])


# # percentage of outliers is more than 5%. So we ignore the outliers for this column.

# In[77]:


sns.boxplot(data1.serum_cholesterol_mg_per_dl)


# In[78]:


IQR=stats.iqr(data1.serum_cholesterol_mg_per_dl)
IQR
Q1=data1.serum_cholesterol_mg_per_dl.quantile(.25)
Q3=data1.serum_cholesterol_mg_per_dl.quantile(.75)
maxm=Q3+IQR*1.5
minm=Q1-IQR*1.5
minm


# In[79]:


data1.loc[data1['serum_cholesterol_mg_per_dl']>maxm]


# In[80]:


data1.loc[data1['serum_cholesterol_mg_per_dl']>maxm, 'serum_cholesterol_mg_per_dl']=np.median(data1.serum_cholesterol_mg_per_dl)


# In[81]:


sns.boxplot(data1.serum_cholesterol_mg_per_dl)


# In[82]:


sns.boxplot(data1.oldpeak_eq_st_depression)


# In[83]:


IQR=stats.iqr(data1.oldpeak_eq_st_depression, interpolation='midpoint')
IQR
Q1=data1.oldpeak_eq_st_depression.quantile(.25)
Q3=data1.oldpeak_eq_st_depression.quantile(.75)
maxm=Q3+IQR*1.5
minm=Q1-IQR*1.5
maxm
minm


# In[84]:


len(data1.loc[data1['oldpeak_eq_st_depression']>maxm, 'oldpeak_eq_st_depression'])


# In[85]:


percentage_of_outliers=(13*100)/180
print(percentage_of_outliers)
#since this column too have more than 5% outliers, we dont handle the outliers


# # Hence Outliers are handled

# # Fetaure Selection

# In[86]:


## Checking correlation

plt.figure(figsize=(20, 25))#canvas size
sns.heatmap(data1.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":10})#plotting heat map to check correlation


# # Model Creation

# In[87]:


## Creating independent and dependent variable
x = data1.drop('heart_disease_present', axis=1)#dependent variable 
y = data1.heart_disease_present #independent variable 


# In[88]:


x


# In[89]:


y


# ### Scaling the Data

# In[90]:


## scaling the data as all features seems to be near to normal distribution
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()## objet creation
x_scaled = scaler.fit_transform(x)# scaling independent variables


# In[91]:


x_scaled #Scaled


# In[92]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=40)


# In[93]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()#object creation of logistic regression
log_reg.fit(x_train,y_train)#training model with training data


# In[94]:


y_train_pre=log_reg.predict(x_train)# predicting y_train 


# In[95]:


y_train_pre


# ### Let's see how well our model performs on the test data set.

# In[96]:


y_pred = log_reg.predict(x_test) # testing model 


# In[97]:


y_train.shape # to know the shape of y_train (rows and columns)


# In[98]:


y_pred.shape # to know the shape of y_pred (rows and columns )


# In[99]:


## calculating accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report,roc_auc_score
accuracy = accuracy_score(y_train,y_train_pre)# model traning accuracy  
print(accuracy*100,'%')


# In[100]:


test_accuracy=accuracy_score(y_test,y_pred)
print(test_accuracy*100,'%')


# In[101]:


# Precison on testing
Precision = precision_score(y_test,y_pred)# the number of true positive divided by the total number of positive prediction
print(Precision*100,'%')


# In[102]:


# Recall on testing
Recall = recall_score(y_test,y_pred)# the total number of positive results how many positives were correctly predicted by the model.
print(Recall*100,'%')


# In[103]:


# F1 Score
F1_Score = f1_score(y_test,y_pred)# when precision and recall both are important
print(F1_Score*100,'%')


# In[104]:


# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc


# In[105]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[106]:


##confusion matrix
pd.crosstab(y_test, y_pred)


# In[107]:


report=classification_report(y_test, y_pred)# it will give precision,recall,f1 scores and accuracy  
print(report)


# # ROC Plot

# In[108]:


## Prediciting the probabilities of class 1
probs=log_reg.predict_proba(x_test)[:,1]


# In[109]:


probs#probabilities of class 1


# In[110]:


## Defining the threshold limit
def predict_threshold (model,X_test,thresholds):
    return np.where(model.predict_proba(X_test)[:,1]>thresholds,1,0)#checking where probability of class 1 is  greater than threshold


# In[111]:


import numpy as np
from sklearn.metrics import confusion_matrix
for thr in np.arange(0,1.0,0.1):# it will create matrix /array from range 0 to 1 with step 0.1
    y_predict = predict_threshold(log_reg,x_test,thr)# it will check result  for  each threshold from 0 to 0.1
    print("Threshold :",thr)#printing threshold
    print(confusion_matrix(y_test,y_pred))# confusion matrix for each prediction


# In[112]:


## visualizing the roc plot
def plot_roc_curve(fpr, tpr):# function to plot roc curve
    plt.plot(fpr, tpr, color='orange', label='ROC')#line plot between fpr and tpr
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')# assigning name to  x axis
    plt.ylabel('True Positive Rate')# assigning name to y axis
    plt.title('Receiver Operating Characteristic (ROC) Curve')#assigning name to curve  
    plt.legend()#area describing the elements of the graph
    plt.show()#to show graph without location


# In[113]:


from sklearn.metrics import roc_auc_score,roc_curve ## used to compare multiple models
auc = roc_auc_score(y_test, probs) #roc curve 
print('AUC: %.2f' % auc)


# In[114]:


fpr, tpr, thresholds = roc_curve(y_test, probs)
# it will return 
#Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
#Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
#Decreasing thresholds on the decision function used to compute fpr and tpr


# In[115]:


import matplotlib.pyplot as plt
plot_roc_curve(fpr, tpr)#plotting ruc curve


# In[125]:


## Balacing the data                    
from collections import Counter
from imblearn.over_sampling import SMOTE# importing counter to check count of each label
sm=SMOTE()
print(Counter(y))
X_sm,y_sm=sm.fit_resample(x_train,y_train)
print(Counter(y_sm))


# In[126]:


from sklearn.tree import DecisionTreeClassifier #importing decision tree from sklearn.tree
dt=DecisionTreeClassifier()
dt.fit(X_sm,y_sm)
y_hat=dt.predict(x_test)
y_hat


# In[127]:


from sklearn.tree import DecisionTreeClassifier #importing decision tree from sklearn.tree
dt=DecisionTreeClassifier() #object creation for decision tree  
dt.fit(x_train,y_train) #training the model


# In[128]:


y_train_predict=dt.predict(x_train)#predicting training data to check training performance 
y_train_predict


# In[129]:


y_test_pred=dt.predict(x_test) #prediction
y_test_pred #predicted values for testing


# In[130]:


## Evalauting the model
from sklearn.metrics import accuracy_score,classification_report,f1_score#importing mertics to check model performance
##Training score
y_train_predict=dt.predict(x_train)#passing X_train to predict Y_train
acc_train=accuracy_score(y_train,y_train_predict)#checking accuracy
print(acc_train*100,'%')


# In[131]:


print(classification_report(y_train,y_train_predict))


# In[132]:


pd.crosstab(y_train,y_train_predict)


# In[133]:


## test acc
test_acc=accuracy_score(y_test,y_test_pred)#testing accuracy 
print(test_acc*100,'%')


# In[134]:


## test score
test_f1=f1_score(y_test,y_test_pred)#f1 score
print(test_f1*100,'%')


# In[135]:


print(classification_report(y_test,y_test_pred))


# In[136]:


pd.crosstab(y_test,y_test_pred)


# In[137]:


from sklearn.model_selection import GridSearchCV
#It helps to loop through predefined hyperparameters and fit your estimator (model) on your training set. 
#So,in the end, you can select the best parameters from the listed hyperparameters.


# In[138]:


from sklearn.ensemble import RandomForestClassifier#importing randomforest

rf_clf = RandomForestClassifier(n_estimators=100)#object creation ,taking 100 decision tree in random forest 
rf_clf.fit(x_train,y_train)#training the data


# In[139]:


from sklearn.ensemble import RandomForestClassifier


# In[140]:


RandomForestClassifier()


# In[141]:


y_predict=rf_clf.predict(x_test)#testing


# In[142]:


print(classification_report(y_test,y_predict))


# In[143]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=1400, stop=3000, num=10)]#List Comprehension-using for loop in list
max_features = ['auto', 'sqrt']#maximum number of features allowed to try in individual tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]#List Comprehension-using for loop in list
max_depth.append(None)
min_samples_split = [2, 5, 10]#minimum number of samples required to split an internal node
min_samples_leaf = [1, 2, 4]#minimum number of samples required to be at a leaf node.
bootstrap = [True, False]#sampling 

#dictionary for hyperparameters
random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf1 = RandomForestClassifier(random_state=42)#model

rf_cv = RandomizedSearchCV(estimator=rf_clf1, scoring='f1',param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)


# In[152]:


rf_clf2 = RandomForestClassifier(n_estimators=1400,min_samples_split=2,min_samples_leaf=1,max_features='auto',max_depth=40,bootstrap=False)#passing best parameter to randomforest
rf_clf2.fit(x_train, y_train)#training 
y_predict=rf_clf2.predict(x_test)#testing
f1_score=f1_score(y_test,y_predict)#checking performance


# In[153]:


f1_score#calling variable


# In[154]:


print(classification_report(y_test,y_predict))


# In[161]:


x_sm, y_sm = sm.fit_resample(x_train,y_train)


# In[164]:


from collections import Counter
print("Actual Classes",Counter(y))
print("SMOTE Classes",Counter(y_sm))


# In[167]:


# Support Vector Classifier Model

from sklearn.svm import SVC
svclassifier = SVC() ## base model with default parameters
svclassifier.fit(X_sm, y_sm)


# In[168]:


y_pred=svclassifier.predict(x_train)


# In[169]:


from sklearn.metrics import accuracy_score,classification_report,f1_score


# In[170]:


#f1 Score
f1=f1_score(y_train,y_pred)


# In[171]:


print(f1)


# In[172]:


# Predict output for X_test
y_hat=svclassifier.predict(x_test)


# In[173]:


## evaluating the model created
acc=accuracy_score(y_test,y_hat)
acc


# In[174]:


# Classification report measures the quality of predictions. True Positives, False Positives, True negatives and False Negatives 
# are used to predict the metrics of a classification report 
print(classification_report(y_test,y_hat))


# In[ ]:





# In[ ]:





# In[ ]:




