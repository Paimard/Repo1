#!/usr/bin/env python
# coding: utf-8

# # Implementierung des Algorithmus Schritt für Schritt

# ### 1. Use Numpy and Pandas library in Python. All these rich libraries of Python made the machine learning algorithm a lot easier. Import the packages and the dataset:

# In[ ]:


# import pandas as pd
import numpy as np
df = pd.read_csv('DataSet1.txt', header = None)
df.head()


# ### 2. Add a column of ones for the bias term. I chose 1 because if you multiply one with any value, that value does not change:

# In[3]:


df = pd.concat(
              [pd.Series(1, index=df.index, name='00'), df], 
              axis=1)
df.head()


# ### 3. Define the input variables or the independent variables X and the output variable or dependent variable y. In this dataset, columns 0 and 1 are the input variables and column 2 is the output variable.

# In[5]:


X = df.drop(columns=2)
y = df.iloc[:, 3]


# ### 4. Normalize the input variables by dividing each column by the maximum values of that column. That way, each column’s values will be between 0 to 1. 
# 
# **NOTE**: This step is not essential. But it makes the algorithm to reach it’s optimum faster. Also, if you notice the dataset, elements of column 0 are too big compared to the elements of column 1. If you normalize the dataset, it prevents column one from being too dominating in the algorithm.

# In[6]:


for i in range(1, len(X.columns)):
    X[i-1] = X[i-1]/np.max(X[i-1])
X.head()


# ### 5. initiate the theta values. I am initiating them as zeros. But any other number should be alright.

# In[7]:


theta = np.array([0]*len(X.columns))
#Output: array([0, 0, 0])


# ### 6. Calculate the number of training data that is denoted as m in the formula above:

# In[8]:


m = len(df)


# ### 7. Define the hypothesis function

# In[9]:


def hypothesis(theta, X):
    return theta*X


# ### 8. Define the cost function using the formula of the cost function explained above

# In[11]:


def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1=np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*47)


# ### 9. Write the function for the gradient descent. This function will take X, y, theta, learning rate(alpha in the formula), and epochs(or iterations) as input. We need to keep updating the theta values until the cost function reaches its minimum.

# In[12]:


def gradientDescent(X, y, theta, alpha, i):
    J = []  #cost function in each iterations
    k = 0
    while k < i:        
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/len(X))
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta


# ### 10. Use the gradient descent function to get the final cost, the list of cost in each iteration, and the optimized parameters theta. I chose alpha as 0.05. But you can try with some other values like 0.1, 0.01, 0.03, 0.3 to see what happens. I ran it for 10000 iterations. Please try it with more or fewer iterations to see the difference.

# In[13]:


J, j, theta = gradientDescent(X, y, theta, 0.05, 10000)


# ### 11. Predict the output using the optimized theta

# In[14]:


y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)


# ### 12. Plot the original y and the predicted output ‘y_hat’

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=list(range(0, 47)),y= y, color='blue')         
plt.scatter(x=list(range(0, 47)), y=y_hat, color='black')
plt.show()


# ### 13. Plot the cost of each iteration to see the behavior

# In[16]:


plt.figure()
plt.scatter(x=list(range(0, 10000)), y=J)
plt.show()


# In[ ]:




