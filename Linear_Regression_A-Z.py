#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[26]:


data=pd.read_csv(Salary_Data.csv')


# In[27]:


data.head()


# In[29]:


data.shape


# In[30]:


target=data['Salary']


# In[31]:


data=data.drop(['Salary'],axis='columns')


# In[32]:


data.head()


# In[33]:


plt.scatter(data,target,color='blue',marker='*')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[34]:


from sklearn.model_selection import train_test_split


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=1/3,random_state=0)


# In[49]:


len(x_train)


# In[50]:


from sklearn.linear_model import LinearRegression


# In[51]:


model=LinearRegression()


# In[52]:


model.fit(x_train,y_train)


# In[61]:


model.predict([[1.2]])


# In[53]:


model.score(data,target)


# In[57]:


plt.plot(x_train,model.predict(x_train),color='blue',marker='*')
plt.scatter(x_train,y_train,color='red')


# In[56]:


plt.plot(x_test,model.predict(x_test),color='blue',marker='*')
plt.scatter(x_test,y_test,color='red')


# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[47]:


regressor.score(X_test,y_test)


# In[60]:


regressor.predict([[1.2]])


# In[ ]:




