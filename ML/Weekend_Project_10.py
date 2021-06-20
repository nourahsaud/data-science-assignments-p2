#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/gumdropsteve/intro_to_machine_learning/blob/main/day_02/02_assignment_cost.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().run_cell_magic('capture', '', '# STARTER CODE - RUN THIS CELL - DO NOT CHANGE\n!pip install category_encoders\nimport seaborn as sns\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport category_encoders as ce\nfrom sklearn.linear_model import LinearRegression, LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, accuracy_score, recall_score, precision_score\ndf_r = sns.load_dataset("tips").dropna()\ndf_c = sns.load_dataset("titanic").dropna()\nmodel_r = LinearRegression()\nmodel_c = LogisticRegression()\nx_train_r, x_test_r, y_train_r, y_test_r = train_test_split(df_r.drop("total_bill", axis=1), df_r["total_bill"])\nx_train_c, x_test_c, y_train_c, y_test_c = train_test_split(df_c.drop(["survived", "alive", "adult_male"], axis=1), df_c["survived"])\npipe_r = make_pipeline(ce.OrdinalEncoder(), StandardScaler(), LinearRegression()).fit(x_train_r, y_train_r)\npipe_c = make_pipeline(ce.OrdinalEncoder(), StandardScaler(), LogisticRegression()).fit(x_train_c, y_train_c)\ny_pred_r = pipe_r.predict(x_test_r)\ny_pred_c = pipe_c.predict(x_test_c)')


# In[2]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# # Weekend Project 10

# # Heuristic Model for Tips Dataset

# ### Splitting the dataset

# In[3]:


# Spliting the data
train, test = train_test_split(
    df_r,
    test_size=0.2,
    train_size = 0.8,
    random_state=12
)


# #### Read in and Explore the Data

# In[4]:


# data info 
train.info()


# In[5]:


# data description 
train.describe(include="all")


# In[6]:


# exploring the data - view a sample of the dataset to get an idea of the variables
train.sample(5)


# <h4> Describing the columns: </h4>
# <ol>
#     <li> total_bill: total amount of the bill </li>
#     <li> tip: amount of the tip </li>
#     <li> sex: female , male </li>
#     <li> smoker: Yes, No </li>
#     <li> day: day </li>
#     <li> time: Lunch, Dinner </li>
#     <li> size: Number of people </li>
# </ol>

# ### Build the Heuristic model

# #### EDA 's 

# In[7]:


# ploting boxplot for the numerical variables 
plt.figure(figsize=(10,7))
sns.boxplot(data=train)


# ##### Dropping the outliers

# In[10]:


# Total bill
# First quartile (Q1)
Q1 = np.percentile(train['total_bill'], 25, interpolation = 'midpoint')
  
# Third quartile (Q3)
Q3 = np.percentile(train['total_bill'], 75, interpolation = 'midpoint')
  
# Interquaritle range (IQR)
IQR = Q3 - Q1

# lower bound outliers --> Q1 - 1.5(IQR)
# higher bound outliers --> Q3 + 1.5 (IQR)


print(round(Q3+ 1.5*(IQR),4))


# In[11]:


#Dropping the outliers from Total bill column
train=train.drop(train[train['total_bill'] > 37.325 ].index)


# In[12]:


# Tips 
# First quartile (Q1)
Q1 = np.percentile(train['tip'], 25, interpolation = 'midpoint')
  
# Third quartile (Q3)
Q3 = np.percentile(train['tip'], 75, interpolation = 'midpoint')
  
# Interquaritle range (IQR)
IQR = Q3 - Q1

# lower bound outliers --> Q1 - 1.5(IQR)
# higher bound outliers --> Q3 + 1.5 (IQR)


print(round(Q3+ 1.5*(IQR),3))


# In[13]:


#Dropping the outliers from Tip column
train = train.drop(train[train['tip'] > 5.75 ].index)


# In[14]:


# ploting boxplot for the numerical variables (after dropping the outliers)
plt.figure(figsize=(10,7))
sns.boxplot(data=train)


# ##### Baseline model 

# In[21]:


# Baseline model - (Mean of target)
# Create function to perform baseline model 
def baseline_total_bill(df):
    mean = []
    for x in range(len(df)):
        mean.append(19.541)
    
    return mean  


# In[22]:


# Apply Baseline (training data)
train["mean"] = baseline_total_bill(train)
train.head()


# In[23]:


# computing the mse, mae, and rms for training data - baseline model 
total_bill(train, 'mean')


# In[24]:


# Apply Baseline (testing data)
test["mean"] = baseline_total_bill(test)

# GETING THE MSE, MAE, AND RMS for test data - baseline model 
total_bill(test, 'mean')


# In[15]:


#Looking to the histogram for target variable 
px.histogram(train, x = 'total_bill', title='Total bill count', color_discrete_sequence=['#488AC7','#387C44'])


# In[16]:


# Tips based on gender 
px.histogram(train,x="sex", y="tip", title="Tips based on Gender", color_discrete_sequence=['#488AC7','#387C44'])


# In[17]:


# barplot for total bill and day besid on gender
px.histogram(train,x="day", y="total_bill", color="sex",
             barmode="group", title="The Relationship between Total Bill and Day based on Gender",
             color_discrete_sequence=['#488AC7','#387C44'])


# In[18]:


# barplot for total bill and day based on time

px.histogram(train,x="day", y="total_bill", color="time",
             barmode="group", title="The Relationship between Total Bill and Day based on Time",
            color_discrete_sequence=['#488AC7','#387C44'])


# In[19]:


#looking to the correlation between variables
sns.pairplot(train)


# ###### We can see clearly that there is strong realtionship between tip and total bill

# #### Heuristic model

# #### Conditions
# 1. If the tip is >= 5 and tip <= 10 then predict total_bill = 35 
# 2. If the day was sun or sat and sex was male and time was dinner then predict total_bill = 30 
# 3. Otherwise predict the total_bill = 19.541 (The mean of the total_bill)

# In[25]:


# Create function to perform our heuristic
def heuristic_total_bill(df):

    pred = []
    for x in range(len(df)):
        
        if (df.iloc[x]["tip"] >= 5) & (df.iloc[x]["tip"] <= 10) :
                pred.append(35)
                
        elif (df.iloc[x]["day"] == ["sun", "sat"]) & (df.iloc[x]["sex"] == "Male") & (df.iloc[x]["time"] == "Dinner"):
               pred.append(30)
            
        else:
                pred.append(19.541)
    
    return pred


# In[26]:


# Apply Heuristic (training data)
train["pred"] = heuristic_total_bill(train)

train.tail()


# ### Cost function

# In[27]:


# Calculate mse, mae, and rms
def total_bill (df, pred_col):
    
    actual = df["total_bill"]
    if pred_col == 'mean':
        Prediction = df["mean"]
    else:
        Prediction = df["pred"]
    
    mse = mean_squared_error(actual, Prediction)
    mae = mean_absolute_error(actual, Prediction)
    rms = mean_squared_error(actual, Prediction, squared=False)
    
    return mse, mae, rms


# In[28]:


# Computing the mse, mae, and rms for training data
total_bill(train,'pred')


# ### Test the HA model

# In[29]:


# Apply Heuristic (testing data)
test["pred"] = heuristic_total_bill(test)

test.head()


# In[30]:


# GETING THE MSE, MAE, AND RMS for test data
total_bill(test,'pred')


# ### Linear Regression

# #### Regression Cost Functions
# 
# - Use `y_test_r` as your true labels
# - Use `y_pred_r` as your predicted labels

# In[32]:


# Cost Functions for the Linear Regression (Tips)
print("MSE: ", mean_squared_error(y_test_r, y_pred_r))
print("MAE: ",mean_absolute_error(y_test_r, y_pred_r))
print("RMSE: ",mean_squared_error(y_test_r, y_pred_r, squared=False))


# ### conclusion
# 
# <br>
# <center> The smaller the Mse,Mae, and Rms, the closer you are on finding the line of best fit. </center>
# <br>
# 
#                    MSE             MAE            RMS
# #### train - baseline (46.73769934254145, 5.783110497237569, 6.836497593252077)
# #### test - baseline (80.26722140816327, 6.698224489795919, 8.959197587293366)
# 
# #### train - HA (48.901190662983424, 5.78724861878453, 6.9929386285726425)
# #### test - HA (56.235732673469386, 5.587367346938776, 7.499048784577241)
# 
# 
# #### LR (42.304530007800786, 4.802406810159396, 6.504193263410981)
# 
# 
# 

# # Heuristic Model for Titanic Dataset

# ### Splitting the dataset

# In[33]:


#Spliting the data
train, test = train_test_split(
    df_c,
    test_size=0.2,
    train_size = 0.8,
    random_state = 10
)


# #### Read in and Explore the Data

# In[34]:


# data info 
train.info()


# In[35]:


# data description 
train.describe(include="all")


# In[36]:


# exploring the data - view a sample of the dataset to get an idea of the variables
train.sample(5)


# <h4> Describing the columns: </h4>
# <ol>
#     <li> survived: 0 - No , 1 - Yes (Numeric values of alive column's values) </li>
#     <li> pclass (Passenger class): 1 - upper, 2 - middle, 3 - lower (is a proxy for socio-economic status (SES))</li>
#     <li> sex: female , male </li>
#     <li> age: Age </li>
#     <li> sibsp: Number of Siblings/Spouses Aboard </li>
#     <li> parch: Number of Parents/Children Aboard </li>
#     <li> fare: Passenger Fare (British pound) </li>
#     <li> embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) - Abbreviation of embark town column's values </li>
#     <li> class: First, Second, Third </li>
#     <li> who: who is the passenger (woman, man, child) </li>
#     <li> adult_male: True - False </li>
#     <li> deck: first class had the top decks (A-E), second class (D-F), and third class (E-G) </li>
#     <li> alone: True - False </li>
# </ol>

# ### Build the Heuristic model

# #### EDA 's 

# ##### Baseline modle

# In[37]:


# Baseline Model Prediction
# What would be our accuracy if we predicted the majority class

train["survived"].value_counts(normalize=True)


# In[38]:


# Passenger Survival
fig = px.pie(train, names='survived', title='Passenger Survival', hole=0.5, color_discrete_sequence=['#488AC7','#387C44'])
fig.show()


# In[39]:


# survivals based on gender 
px.histogram(train,x="sex", color="alive", barmode="group", 
             title="survivals based on gender", 
             color_discrete_sequence=['#488AC7','#387C44'])


# In[40]:


# survivals based on pclass 
px.histogram(train,x="pclass", color="alive", barmode="group", 
             title="survivals based on Pclass", 
             color_discrete_sequence=['#488AC7','#387C44'])


# In[41]:


# survivals based on age 
px.histogram(train,x="age", color="alive", barmode="group", 
             title="survivals based on Age", 
             color_discrete_sequence=['#488AC7','#387C44'])


# #### Heuristic model

# #### Conditions 
# 1. If the gender (sex) is 'female' then check if the pclss is 1st or 2nd  -- predict survived
# 2. If the gender (sex) is 'male' then check if the pclass is 1st or 2nd, and check the age if its under 12 -- predict -- survived
# 3. Otherwise predict -- died  

# In[42]:


# Heuristic model
def heuristic_survived(df):

    preds = []
    for x in range(len(df)):
        if(df.iloc[x]['sex'] == 'female'): # female
            if(df.iloc[x]['pclass'] <= 2): # 1 or 2 
                preds.append(1)
            else:
                preds.append(0)
        else:
            if(df.iloc[x]['pclass'] <= 2) and (df.iloc[x]['age'] < 17):
                preds.append(1)
            else:
                preds.append(0)
            
    return preds


# In[43]:


# Apply Heuristic (training data)
train["preds"] = heuristic_survived(train)

train.head()


# ### Cost functions 

# In[44]:


# Cost functions for the Heuristic Model here
# Calculate Accuracy Precision and recall
def survived (df):
    
    actual = df["survived"]
    Prediction = df["preds"]
    
    accuracy = accuracy_score(actual, Prediction)
    recall = recall_score(actual, Prediction)
    precision= precision_score(actual, Prediction)
    
   
    return accuracy, recall, precision


# In[45]:


# Calculate Accuracy Precision and recall on the train data 
survived(train)


# ### Test the HA model

# In[46]:


# Apply Heuristic (testing data)
test["preds"] = heuristic_survived(test)
test.head()


# In[47]:


# Calculate Accuracy Precision and recall on the test data 
survived(test)


# ### Logistic Regression

# #### Classification Cost Functions
# 
# - Use `y_test_c` as your true labels
# - Use `y_pred_c` as your predicted labels

# In[49]:


print("Recall Score: ",recall_score(y_test_c, y_pred_c))
print("Precision Score: ",precision_score(y_test_c, y_pred_c))
print("Accuracy Score: ",accuracy_score(y_test_c, y_pred_c))


# ### Concludion 
# 
# 1. Baseline: 1    0.668966 , 0    0.331034
# 
#             Accuracy Score    Recall Score   Precision Score   
# 1. Train: (0.7448275862068966, 0.6494845360824743, 0.9545454545454546)
# 1. Test: (0.8648648648648649, 0.8461538461538461, 0.9565217391304348)
# 
# 1. LR. (0.8214285714285714, 0.7666666666666667, 0.7391304347826086)

# In[ ]:




