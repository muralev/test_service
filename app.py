import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import fasttext


from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

application = Flask(__name__)


#загружаем модели из файла
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
#model = lgb.Booster(model_file='./models/lgbm_model.txt')
model = fasttext.load_model("./models/fastText_model.txt")


# тестовый вывод
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response

# предикт категории
#{"user_message":"example123rfssg gsfgfd"}
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        
        #напишите прогноз и верните его в ответе в параметре 'prediction'
        usr_msg = json_params['user_message']
        if (usr_msg is None) or (usr_msg == ''):
            resp['message'] = str('error!')
            resp['category'] = str('Empty message.')
        else:    
            category = model.predict(usr_msg,k=3)[1]
            category = category.tolist()
            resp['category'] = category

        
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)
