import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

import seaborn as sns
import os
import re
import nltk
import seaborn as sns
from exp import df
from exp import ef
print(list(df))
df['Text'] = df[('Text',)].str.lower()
df=df.drop(('Text',),axis=1)
print(df)
print(list(df))
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
header=['Text']

data=pd.DataFrame(index=df.index,columns=header)
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
for index,row in df.iterrows():
    filter_sentence=' '
    sentence=row['Text']
    sentence=str(sentence)
    sentence=sentence.lower()
    sentence=re.sub(r'[^\w\s]','',sentence)

    words=nltk.word_tokenize(sentence)

    words=[w for w in words if not w in stop_words]
    for words in words:
        filter_sentence=filter_sentence+' '+str(lemmatizer.lemmatize(words))

    data.loc[index,'Text']=filter_sentence

print(df)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv=TfidfVectorizer()
df=df.dropna()
print(df.isnull().sum())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(data['Text'], df['label'], test_size=0.3, random_state=0)

X_traincv=cv.fit_transform(X_train)

X_testcv=cv.transform(X_test)
print(X_traincv.toarray())
print(cv.get_feature_names())


from sklearn.linear_model import LogisticRegression
logic=LogisticRegression()
logic.fit(X_traincv,y_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
pred=logic.predict(X_testcv)
print(accuracy_score(y_test,pred))

from sklearn.pipeline import Pipeline
import joblib
from sklearn import linear_model
pipeline=Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('elf',linear_model.LogisticRegression()),
      ])
pipeline.fit(X_train,y_train)
print(pipeline.predict(['cbse cance and postpone the exam due to covid-19']))

input = 'COVID-19 vaccine registration opens for those 18+ across India on Wednesday.'

import difflib


for i, line in ef.iterrows():
    line = str(line)
    line2=line.lower()
    input=input.lower()
    Sequence = difflib.SequenceMatcher(None, input, line).ratio()
    if Sequence >= 0.30:
        print(line2)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
axes=sns.heatmap(cm,square=True,annot=True, fmt='d',
                  cbar=True, cmap=plt.cm.GnBu)

filename='pipeline.sav'
joblib.dump(pipeline,filename)
loaded_model=joblib.load(filename)