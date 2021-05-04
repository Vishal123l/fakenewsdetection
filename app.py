from flask import Flask, abort, jsonify, request, render_template
import joblib
import pandas as pd
import json
from IPython.display import HTML
import difflib
from exp import mf
import re
import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


pipeline = joblib.load('pipeline.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api',methods=['POST'])
def get_delay():

    result = request.form
    query=request.form['Title']
    result=query

    headings=('Recommended news',)






    list=[]
    list2=[]
    date=[]
    for i,line in mf.iterrows():
          result=str(result)
          result=result.lower()
          line=line['Text']
          line=str(line)
          line = line.lower()
          Seq=difflib.SequenceMatcher(None,result,line).ratio()
          if Seq >=0.50:
                pred = pipeline.predict([result])
                print(pred)
                if pred==0:
                     res_val=" Fake news "
                else:
                     res_val=" Real news "
                return render_template('real.html',prediction_text='This is {}'.format(res_val))

    for i, line in mf.iterrows():
              result = str(result)
              result = result.lower()
              lin = line['Text']
              lin = str(lin)
              lin = lin.lower()
              line2=line['Editor']
              Seq = difflib.SequenceMatcher(None, result, lin).ratio()

              if Seq >= 0.30:
                   print(lin)
                   print(line2)
                   list2.append(line2)
                   list.append(lin)
                   date.append("4 May 2021")



    if len(list)!=0:
              pd.set_option('display.max_colwidth', None)
              dff = pd.DataFrame({"Recommended news": list,"Editor":list2,"Date":date})
              return render_template('real4.html', tables=[dff.to_html(classes='data')], titles=dff.columns.values)
    else:
              return render_template('real3.html')


if __name__ == '__main__':
    app.run(port=8080, debug=True)
