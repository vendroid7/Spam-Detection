# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:28:52 2021

@author: Venkatesh Iyer
"""



import numpy as np
import pickle as pkl
import pandas as pd
import flasgger
from flasgger import Swagger


clf=pkl.load((open('classifier.pkl','rb' )))
tfidf=pkl.load((open('tfidf.pkl','rb')))


from flask import Flask,request


app = Flask(__name__)
Swagger(app)

@app.route('/')
def hello_world():
    return 'Hello, World,v!'


@app.route('/get_text',methods=["Get"])
def get_text():
    
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: text
        in: query
        type: string
        required: true
        
    responses:
        200:
            description: The output values
     """
        
    text=request.args.get('text')
    text=tfidf.transform([text])
    prediction=clf.predict(text.toarray())
    return "prediction is "+str(prediction[0])


if __name__ == "__main__":
	app.run()
    