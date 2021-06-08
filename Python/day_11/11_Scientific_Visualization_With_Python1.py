#!/usr/bin/env python
# coding: utf-8

# <h1><center><font color='#3D3C3A'> Handling missing values </font></center></h1>
# <img src="missing_values.png">
# 
# [Dataset](https://drive.google.com/file/d/1TjvAtdw2mz5zKDYJMBUFHqjOhey8u95q/view)
# 

# <center><font color='#3D3C3A'> A missing value is a value that is left blank or filled with another value to represent the value is missing. </font></center>

# <h3><font color='#3D3C3A'> Assignment: </font></h3>
# <ul>
#     <li> <font color='#3D3C3A'> Create a histogram or bar chart of what the distribution of values looks like before handling NaN values and after. </font> </li>
#     <li> <font color='#3D3C3A'> Determine how many missing values are present in each column </font> </li>
#     <li> <font color='#3D3C3A'> Determine the total amount of missing values </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and replace NaN values with the mean </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and replace NaN values with the median </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and replace NaN values with the mode </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and replace NaN values with your own value </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and use backwards filling </font> </li>
#     <li> <font color='#3D3C3A'> Choose a column and use forwards filling </font> </li>
#     </ul>

# In[555]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# In[556]:


# Reading dataset
PATH = '/Users/nourahalsaadan/Desktop/Coding_Dojo/Python/day_11/burritos.csv'
df = pd.read_csv(PATH)


# In[557]:


# Visually inspec first 5 rows
df.head()


# In[558]:


# Get the number of entries and columns
df.shape


# In[559]:


# df summary
df.info()


# In[560]:


# Computes a summary of statistics pertaining to df columns.
df.describe()


# In[561]:


# Shows a summary of categorical objects in df columns.
df.describe(exclude= 'number')


# In[562]:


# Count total NaN at each column 
df.isnull().sum()


# In[563]:


# Count total NaN at each row 
for i in range(len(df.index)) :
    print('Total NaN in row', i + 1, ':',
          df.iloc[i].isnull().sum())


# In[564]:


# Count total NaN 
print('\nCount total NaN in a DataFrame : \n\n',
       df.isnull().sum().sum())


# <h1><font color='#3D3C3A'>Bar plot before handling missing values</font></h1>

# In[565]:


# Look at total NaN values per column graphically
from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi = 100)
df.isnull().sum().sort_values().plot(kind='bar', title= 'Missing Values (Shape: (424, 66))');
plt.xlabel('Columns')
plt.ylabel('Missing Valuses')


# In[566]:


# Dropping rows only where all values are NaN values
# It looks like there are no rows where all values are NaN
df.dropna(axis = 'index', how = 'all', inplace = True)
df.shape


# ~ There are no rows where all values are NaN

# In[567]:


# Dropping columns only where all values are NaN values
df.dropna(axis = 'columns', how = 'all', inplace = True)
df.shape


# ~ There was one column that has all its' values NaN ~ <font color='#C24641'> ' Queso ' </font>

# <h2><font color='#3D3C3A'> Replace NaN values in the Yelp column with the mean </font></h2>

# In[568]:


# Get uniqe values in the Yelp column
df['Yelp'].unique()


# In[569]:


# Count NaN values in the Yelp column 
df['Yelp'].isnull().sum()


# In[570]:


# Plot a histogram for the Yelp column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Yelp'].plot(kind="hist", title="Histogram of Yelp column before handling NaN values");


# In[571]:


# Calculate the mean
mean = round(df['Yelp'].mean(),1)
mean


# In[572]:


# Replace NaN values with the mean
df['Yelp'] = df['Yelp'].fillna(mean)
df['Yelp'].isnull().sum()


# In[573]:


# Plot a histogram for the Yelp column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Yelp'].plot(kind="hist", title="Histogram of Yelp column after handling NaN values");


# <h2><font color='#3D3C3A'> Replace NaN values in the Length column with the median </font></h2>

# In[574]:


# Get uniqe values in the Length column
df['Length'].unique()


# In[575]:


# Count NaN values in the Length column 
df['Length'].isnull().sum()


# In[576]:


# Plot a histogram for the Length column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Length'].plot(kind="hist", title="Histogram of Length column before handling NaN values");


# In[577]:


# Calculate the median
median = round(df['Length'].median(),1)
median


# In[578]:


# Replace NaN values with median
df['Length'] = df['Length'].replace(np.NaN, median)


# In[579]:


# Plot a histogram for the Length column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Length'].plot(kind="hist", title="Histogram of Length column after handling NaN values");


# <h2><font color='#3D3C3A'> Replace NaN values in the overall column with the mode </font></h2>

# In[580]:


# Get uniqe values in the overall column
df['overall'].unique()


# In[581]:


# Count NaN values in the overall column 
df['overall'].isnull().sum()


# In[582]:


# Plot a histogram for the overall column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['overall'].plot(kind="hist", title="Histogram of overall column before handling NaN values");


# In[583]:


# Calculate mode
mode = df['overall'].mode()[0]
mode


# In[584]:


# Replace NaN values with mode
df['overall'] = df['overall'].replace(np.NaN, mode)


# In[585]:


# Plot a histogram for the overall column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['overall'].plot(kind="hist", title="Histogram of overall column after handling NaN values");


# ~ The change in the plot is not obvious because the overall column have only 2 missing values

# <h2><font color='#3D3C3A'> Replace NaN values in the Mass (g) column with a choosen value </font></h2>
# 

# In[586]:


# Get uniqe values in the Mass (g) column
df['Mass (g)'].unique()


# In[587]:


# Count NaN values in the Mass (g) column 
df['Mass (g)'].isnull().sum()


# In[588]:


# Plot a histogram for the Mass (g) column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Mass (g)'].plot(kind="hist", title="Histogram of Mass (g) column before handling NaN values");


# In[589]:


# choosen value 
unknown = -100


# In[590]:


# Replace NaN values with the choosen value
df['Mass (g)'] = df['Mass (g)'].replace(np.NaN, unknown)


# In[591]:


# Plot a histogram for the Mass (g) column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Mass (g)'].plot(kind="hist", title="Histogram of Mass (g) column after handling NaN values");


# <h2><font color='#3D3C3A'> Replace NaN values in the Cost column useing backwards filling </font></h2>

# In[592]:


# Get uniqe values in the Cost column
df['Cost'].unique()


# In[593]:


# Count NaN values in the Cost column 
df['Cost'].isnull().sum()


# In[594]:


# Plot a histogram for the Cost column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Cost'].plot(kind="hist", title="Histogram of Cost column before handling NaN values");


# In[595]:


# Backwards filling
df['Cost'].fillna(method="bfill", inplace = True)


# In[596]:


# Count NaN values in the Cost column after backward filling
df['Cost'].isnull().sum()


# In[597]:


# Plot a histogram for the Cost column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Cost'].plot(kind="hist", title="Histogram of Cost column before handling NaN values");


# <h2><font color='#3D3C3A'> Replace NaN values in the Hunger column useing backwards filling </font></h2>

# In[598]:


# Get uniqe values in the Cost column
df['Hunger'].unique()


# In[599]:


# Count NaN values in the Cost column 
df['Hunger'].isnull().sum()


# In[600]:


# Plot a histogram for the Hunger column before handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Hunger'].plot(kind="hist", title="Histogram of Hunger column before handling NaN values");


# In[601]:


# Forward filling
df['Hunger'].fillna(method="ffill", inplace = True)


# In[602]:


# Count NaN values in the Hunger column after Forward filling
df['Hunger'].isnull().sum()


# In[603]:


# Plot a histogram for the Hunger column after handling the missing values
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi = 80)
df['Hunger'].plot(kind="hist", title="Histogram of Hunger column after handling NaN values");


# <h1><font color='#3D3C3A'>Bar plot after handling missing values in some colums</font></h1>

# In[604]:


# Look at total NaN values per column graphically
from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi = 100)
df.isnull().sum().sort_values().plot(kind='bar', title= 'Missing Values (Shape: (424, 66))');
plt.xlabel('Columns')
plt.ylabel('Missing Valuses')

