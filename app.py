from flask import Flask, abort, jsonify, request, render_template
import joblib
import json
from feature import *
pipeline = joblib.load('pipeline.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api',methods=['POST'])
def get_delay():

    result=request.form
    query_title = result['Title']
    query_author = result['Author']
    query_text = result['Text']
    query = get_all_query(query_title, query_author, query_text)
    user_input = {'query':query}
    pred = pipeline.predict(query)
    print(pred)
    if pred==0:
      res_val=" Fake news "
    else: 
      res_val=" Real news "
    
    return render_template('real.html',prediction_text='This is {}'.format(res_val))

if __name__ == '__main__':
    app.run(port=8080, debug=True)
