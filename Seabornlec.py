
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


get_ipython().magic(u'matplotlib inline')


# In[3]:


tips = sns.load_dataset("tips")


# In[4]:


tips.head()


# In[6]:


sns.distplot(tips["total_bill"], bins = 30)


# In[8]:


sns.jointplot(x= "total_bill", y= "tip", data= tips, kind= "hex")


# In[10]:


sns.pairplot(tips, hue = "sex")


# In[11]:


sns.rugplot(tips["total_bill"])


# In[15]:


import numpy as np


# In[16]:


sns.barplot(x= "sex", y= "total_bill", data= tips, estimator = np.std)


# In[17]:


sns.countplot(x= "sex", data = tips)


# In[19]:


sns.boxplot(x= "day", y= "total_bill", data= tips, hue = "smoker")


# In[21]:


sns.violinplot(x= "day", y= "total_bill", data= tips, hue = "sex", split= True)


# In[23]:


sns.stripplot(x= "day", y= "total_bill", data= tips, jitter = True)


# In[24]:


sns.swarmplot(x= "day", y= "total_bill", data= tips)


# In[25]:


sns.factorplot(x= "day", y= "total_bill", data=tips, kind= "bar")


# In[29]:


tc= tips.corr()


# In[30]:


tc


# In[33]:


sns.heatmap(tc, annot= True)


# In[26]:


flights = sns.load_dataset("flights")


# In[27]:


flights.head()


# In[35]:


fp = flights.pivot_table(index = "month", columns= "year", values = "passengers")


# In[36]:


fp


# In[38]:


sns.heatmap(fp, annot= True, cmap= "magma")


# In[39]:


sns.clustermap(fp)


# In[42]:


sns.lmplot(x= "total_bill", y= "tip", data= tips, hue = "sex", markers= ["o", "v"])


# In[44]:


sns.lmplot(x= "total_bill", y= "tip", data = tips, col= "sex", row = "time")


# In[45]:


iris = sns.load_dataset("iris")


# In[46]:


iris.head()


# In[47]:


iris["species"].unique()


# In[48]:


sns.pairplot(iris)


# In[51]:


grid_iris = sns.PairGrid(iris)


# In[54]:


sns.set_style("darkgrid")
sns.countplot(x= "sex", data = tips)


# In[56]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[57]:


titanic = sns.load_dataset("titanic")


# In[58]:


titanic.head()


# In[59]:


sns.jointplot(x= "fare", y= "age", data = titanic)


# In[64]:


sns.distplot(titanic["fare"], kde = False)


# In[65]:


sns.boxplot(x= "class", y= "age", data = titanic)


# In[67]:


sns.swarmplot(x= "class", y = "age", data= titanic)


# In[69]:


sns.countplot(x= "sex", data = titanic)


# In[70]:


titanic.corr()


# In[71]:


sns.heatmap(titanic.corr())
plt.title("Titanic")

