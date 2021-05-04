#! / usr / bin / env
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import csv
from fuzzywuzzy import fuzz, process
import difflib

# In[49]:


df = pd.read_csv(r'D:\HP1\Downloads\timesofindia.csv')
header_names = ['text']
header_names2=['Editor']

# In[50]:


df.columns

# In[51]:


df1 = df['w_tle']
df1

# In[52]:


df1.isnull().sum()

# In[53]:


df1 = df1.dropna()

# In[54]:


header_names = ['Text']
df1
df1 = pd.DataFrame(np.c_[df1], columns=[header_names])
df1
df1.to_csv(r"D:\HP1\Downloads\News.csv")

# In[55]:


df2 = df['w_tle 2']
df2.isnull().sum()
df2 = df2.dropna()
df2 = df2.dropna()
df2 = pd.DataFrame(np.c_[df2], columns=[header_names])
df2.to_csv(r"D:\HP1\Downloads\News.csv")

# In[56]:


df3 = df['w_tle 3']
df3.isnull().sum()
df3 = df3.dropna()
df3 = pd.DataFrame(np.c_[df3], columns=[header_names])

# In[57]:


df4 = df['w_tle 4']
df4.isnull().sum()
df4 = df4.dropna()
df4 = pd.DataFrame(np.c_[df4], columns=[header_names])

# In[58]:


df5 = df['w_tle 5']
df5.isnull().sum()
df5 = df5.dropna()
df5 = pd.DataFrame(np.c_[df5], columns=[header_names])

# In[59]:


df6 = df['w_tle 6']
df6.isnull().sum()
df6 = df6.dropna()
df6 = pd.DataFrame(np.c_[df6], columns=[header_names])

# In[60]:


df7 = df['w_tle 7']
df7.isnull().sum()
df7 = df7.dropna()
df7 = pd.DataFrame(np.c_[df7], columns=[header_names])

# In[61]:


df8 = df['w_tle 2']
df8.isnull().sum()
df8 = df8.dropna()
df8 = pd.DataFrame(np.c_[df8], columns=[header_names])

# In[62]:


df9 = df['w_tle 2']
df9.isnull().sum()
df9 = df9.dropna()
df9 = pd.DataFrame(np.c_[df9], columns=[header_names])

# In[63]:


df10 = df['w_tle 10']
df10.isnull().sum()
df10 = df10.dropna()
df10 = pd.DataFrame(np.c_[df10], columns=[header_names])

# In[64]:


df11 = df['w_tle 11']
df11.isnull().sum()
df11 = df11.dropna()
df11 = pd.DataFrame(np.c_[df11], columns=[header_names])

# In[65]:


df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11])

# In[66]:
a=len(df)
df
a1=[]
for i in range(a):
    a1.append("times of india")
a1=pd.DataFrame(np.c_[a1], columns=[header_names2])
# In[67]:


df.tail(10)

# In[72]:


# In[80]:


bf = pd.read_csv(r'D:\HP1\Downloads\aninews.csv')
bf
bf.columns

# In[81]:


bf

# In[82]:


bf1 = bf['title']
bf1 = bf1.dropna()
bf1 = pd.DataFrame(np.c_[bf1], columns=[header_names])

bf2 = bf['title 2']
bf2 = bf2.dropna()
bf2 = pd.DataFrame(np.c_[bf2], columns=[header_names])

bf3 = bf['title 3']
bf3 = bf3.dropna()
bf3 = pd.DataFrame(np.c_[bf3], columns=[header_names])

bf4 = bf['title 4']
bf4 = bf4.dropna()
bf4 = pd.DataFrame(np.c_[bf4], columns=[header_names])

bf5 = bf['title 5']
bf5 = bf5.dropna()
bf5 = pd.DataFrame(np.c_[bf5], columns=[header_names])

bf6 = bf['title 6']
bf6 = bf6.dropna()
bf6 = pd.DataFrame(np.c_[bf6], columns=[header_names])

bf7 = bf['title 7']
bf7 = bf7.dropna()
bf7 = pd.DataFrame(np.c_[bf7], columns=[header_names])

bf8 = bf['title 8']
bf8 = bf8.dropna()
bf8 = pd.DataFrame(np.c_[bf8], columns=[header_names])

bf9 = bf['title 9']
bf9 = bf9.dropna()
bf9 = pd.DataFrame(np.c_[bf9], columns=[header_names])

bf10 = bf['title 10']
bf10 = bf10.dropna()
bf10 = pd.DataFrame(np.c_[bf10], columns=[header_names])

# In[83]:


bf = pd.concat([bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8, bf9, bf10])
b=len(bf)
# In[84]:
b1=[]

for i in range(b):
    b1.append("Ani")

b1=pd.DataFrame(np.c_[b1], columns=[header_names2])

print(len(bf))

# In[85]:


cf = pd.read_csv(r'D:\HP1\Downloads\economictimes.csv')

# In[86]:


cf.head()

# In[87]:


cf1 = cf['clr']
cf1 = cf1.dropna()
cf1 = pd.DataFrame(np.c_[cf1], columns=[header_names])

cf2 = cf['flr']
cf2 = cf2.dropna()
cf2 = pd.DataFrame(np.c_[cf2], columns=[header_names])

cf3 = cf['clr 2']
cf3 = cf3.dropna()
cf3 = pd.DataFrame(np.c_[cf3], columns=[header_names])

# In[88]:


cf = pd.concat([cf1, cf2, cf3])

# In[89]:


c=len(cf)
c1=[]
for i in range(c):
    c1.append("economic times")
c1=pd.DataFrame(np.c_[c1], columns=[header_names2])
# In[90]:

jf=pd.read_csv(r"D:\HP1\Downloads\ndtv.csv")
Jf=df.dropna()
jf1=jf['newsHdng']
jf1 =pd.DataFrame(np.c_[jf1], columns=[header_names])
d=len(jf1)
print(len(jf1))
j1=[]
for i in range(d):
    j1.append("ndtv")
d1 =pd.DataFrame(np.c_[j1], columns=[header_names2])


sf=pd.concat([a1,b1,c1,d1])
sf=sf.reset_index(drop=True)



df = pd.concat([df, bf, cf,jf1], axis=0)

# In[91]:
ef=df.copy()
ef=ef.reset_index(drop=True)
df

# In[92]:


df.isnull().sum()

# In[93]:


# In[94]:


df = df.reset_index(drop=True)
print(df)
# In[95]:


df.head()

# In[96]:




# In[97]:



# In[98]:


df.columns

# In[99]:


input = 'Weekend curfew likely in Delhi amid Covid surge.'
for i, line in df.iterrows():
    line = str(line)
    Sequence = difflib.SequenceMatcher(None, input, line).ratio()
    if Sequence >= 0.40:
        print(Sequence)
        print(line)

# In[100]:


line = 'Voters vexed as poll booths.'
Sequence = difflib.SequenceMatcher(None, input, line).ratio()
print(Sequence)

# In[ ]:
mf=pd.concat([ef,sf],axis=1)


mf=mf.reset_index(drop=True)
print(mf)

print(sf)
fake = pd.read_csv(r'D:\HP1\Downloads\Fake.csv')
fake
fake1 = fake[1:100]
fake2 = fake[100:200]
fake = fake.dropna()
fake = pd.concat([fake1, fake2])
fake = fake['title']
fake = pd.DataFrame(np.c_[fake], columns=[header_names])
df = pd.concat([df,fake], ignore_index=True)

list=[]
for i in range(0,len(df)):
    if i<=352:
        list.append(1)
    else:
        list.append(0)
label=pd.DataFrame({'label':list})
df=pd.concat([df,label],axis=1)
df=df.dropna()
print(df)









































