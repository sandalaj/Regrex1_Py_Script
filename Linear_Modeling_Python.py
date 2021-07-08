#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling - Python

# ### Import required packages

# In[3]:

import sys
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ### Read in the data

# In[4]:

print(sys.argv[1])
dataset = pd.read_csv(sys.argv[1])
print(dataset)


# ### Plot the scatterplot

# In[11]:


dataset.plot(x='x', y='y', style='o')  
plt.title('Scatterplot - Python')  
plt.xlabel('x')  
plt.ylabel('y')  
plt.show()
plt.savefig("Scatterplot_Py.png")


# ### Divide the data into 'attributes' and 'labels':

# In[6]:


x = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)


# ### Fit the regression

# In[7]:


regressor = LinearRegression()  
regressor.fit(x, y)


# ### Make predictions on the data

# In[8]:


y_pred = regressor.predict(x)


# ### Re-plot the scatterplot with the linear model

# In[10]:


plt.scatter(x, y,  color='gray')
plt.plot(x, y_pred, color='red', linewidth=2)
plt.title('Linear Model - Python')
plt.show()
plt.savefig("Scatterplot_LM_Py.png")

