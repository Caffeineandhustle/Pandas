#!/usr/bin/env python
# coding: utf-8

# # Pandas full tutorial : Most useful techniques

# In[2]:


#import all the libraries 
#Pandas :  for data analysis
#numpy : for Scientific Computing.
#matplotlib and seaborn : for data visualization
#scikit-learn : ML library for classical ML algorithms
#math :for mathematical functions

import pandas as pd
import numpy as np
from numpy.random import randn
from pandas import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

    
   
    


# In[2]:


# To get the version of the pandas you are using

pd.__version__


# In[12]:


#Create Series

l_one= ['a','b','c','d','e']
s= Series(randn(5), index=l_one)
s


# In[13]:


#Check if 'b' is in s 
'b' in s


# In[16]:


#Get the value of index b
s['b']


# In[17]:


#List the indexes in the series

s.index


# In[18]:


#Map series to dictionary

mapping=s.to_dict()
mapping


# In[19]:


#Convert dictionary to series

s=Series(mapping)
s


# In[20]:


#Access values by slicing

s[:3]


# In[22]:


#Convert series to string

string1=s.to_string()
string1


# In[23]:


#Get the index of the series
s.index


# In[25]:


#Map new indexes to a series

s=Series(mapping, index=['a','b','c','g','h'])
s


# In[27]:


#Check for Null values

s[isnull(s)]


# In[28]:


#Check for not null values

s[notnull(s)]


# In[32]:


#Drop the rows where at least one element is missing

s.dropna()


# In[33]:


#Arithmatic operation ( +,-,*,/)
s*2


# In[34]:


s+2


# # Data frame : 2 D collection of the series
# 

# In[11]:


#Build a data frame

df=DataFrame({'a': np.random.randn(6),'b': ['hi','bye']*3, 
             'c': np.random.randn(6) })
df


# In[38]:


#List columns of the dataframe
df.columns


# In[39]:


#List rows of the dataframe
df.index


# In[43]:


#List values in the column a
df['a']


# In[44]:


#Add another column to the existing dataFrame
df['d']= ['whats up']*6


# In[45]:


#Print the dataFrame
df


# In[46]:


#List values after slicing indexes
df[:4]


# In[15]:


#List values based on label based indexing
df.loc[2:5,['b','c']]


# In[16]:


#List values based on position based indexing
df.iloc[2:5]


# In[17]:


#List values based on some condition 

df[df['c']>0]


# In[55]:


#New DataFrame with index as date

df=DataFrame({'a': np.random.randn(6),'b': ['hi','bye']*3, 
             'c': np.random.randn(6)}, index = pd.date_range('1/1/2020', periods=6) )
df


# In[58]:


#Add more columns to the dataFrame
df=DataFrame({'a': np.random.randn(6),'b': ['hi','bye']*3, 
             'c': np.random.randn(6)},columns=['a','b','c','d'], index = pd.date_range('1/1/2020', periods=6) )
df


# In[59]:


#Check if the values are null or not

isnull(df)


# # Creation from nested Dict

# In[61]:



data={}
for col in ['hi', 'hello', 'hey']:
    for row in ['a','b','c','d']:
        data.setdefault(col,{})[row]=randn()
data        


# In[62]:


#Convert into a dataframe
DataFrame(data)


# In[63]:


#Delete values based on [column][row]

del data['hey']['c']


# In[64]:


DataFrame(data)


# # Data Analysis 

# In[3]:


#Read csv
stocks_data= pd.read_csv('/Users/priyeshkucchu/Desktop/Stocks.csv',engine='python',index_col=0,parse_dates=True)


# In[90]:


#Show data
stocks_data


# In[82]:


#Show top 5 records

stocks_data.head(5)


# In[83]:


#Show last 5 records

stocks_data.tail(5)


# In[71]:


#Show unique values in a column
stocks_data.Name.unique()


# In[84]:


#Show statistics ( mean, std, count, percentile, max, min)
stocks_data.describe()


# In[85]:


#Show column information ( count, type, null or non-null, memory usage)
stocks_data.info()


# In[86]:


#Show no of records in each column
stocks_data.count()


# In[129]:


#s1=stocks_data.Name["AAPL"][-20:]
#s2=stocks_data.Name["AAPL"][-25:-10]
#side_by_side(s1,s2)


# In[19]:


#List values based on position based indexing
df=stocks_data.iloc[1:20]
df


# In[101]:


#Sort values by a column and reset index

stocks_data.sort_values(by='close',ascending=True).reset_index().head(5)


# In[147]:


# Group by a column

stocks_data.groupby(stocks_data.Name=="AAPL").count()


# In[142]:


#Check if the values are null or non-null

stocks_data.isnull().head()


# In[165]:


# Show no of records of a column 
df.Name.count()


# In[169]:


df.shape[0]


# In[9]:


#Boolean Indexing
stocks_data.loc[(stocks_data["open"]>14.0) & (stocks_data["close"]<15.0) & (stocks_data["Name"]=="AAL"),("open","close","Name")] 


# In[17]:


#Apply Function

def missing_values(x):
    return sum(x.isnull())

print("Missing values each column:")
print (stocks_data.apply(missing_values,axis=0))

print("\n Missing values each row:")
print (stocks_data.apply(missing_values,axis=1).head())


# In[24]:


#Imputing missing files

from scipy.stats import mode
mode(stocks_data["open"])


# In[25]:


mode(stocks_data["high"])


# In[26]:


mode(stocks_data["low"])


# In[28]:


stocks_data["open"].fillna(mode(stocks_data["open"]).mode[0],inplace=True)
stocks_data["high"].fillna(mode(stocks_data["high"]).mode[0],inplace=True)
stocks_data["low"].fillna(mode(stocks_data["low"]).mode[0],inplace=True)


# In[29]:


print("Missing values each column:")
print (stocks_data.apply(missing_values,axis=0))


# In[4]:


#Pivot Table

imput_grps= stocks_data.pivot_table(values=["volume"], index=["open","close","high"],aggfunc=np.mean)
print (imput_grps)


# In[22]:


stocks_data.hist(column="open", by="Name", bins=30, figsize=(100,100))


# In[ ]:




