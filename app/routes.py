from flask import current_app as app
from flask import render_template, request
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras import backend as K
import os
from celery import Celery  # Assuming Celery is installed (`pip install celery`)

app.config.update({
    'CELERY_BROKER_URL': 'redis://localhost:6379/0',  # Configure Celery broker (replace if needed)
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0'  # Configure Celery result backend (replace if needed)
})

celery = Celery(app.name)
celery.conf.update(app.config)


SKIN_CLASSES = {
  0: 'Actinic keratosis', #Viêm giác mạc
  1: 'Basal Cell Carcinoma', # Ung thư gia
  2: 'Benign Keratosis',  #Viêm giac mạc bã nhờn
  3: 'Dermatofibroma', # Viêm gia cơ địa
  4: 'Melanoma',      # Khối u ác tính
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion' # Tổn thương mạch máu ở da
}

SKIN_CLASSES_LINK = {
  0: 'https://dermnetnz.org/topics/actinic-keratosis', #Viêm giác mạc
  1: 'https://dermnetnz.org/topics/basal-cell-carcinoma', # Ung thư gia
  2: 'https://dermnetnz.org/topics/seborrhoeic-keratosis',  #Viêm giac mạc bã nhờn
  3: 'https://dermnetnz.org/topics/dermatofibroma', # Viêm gia cơ địa
  4: 'https://dermnetnz.org/topics/melanoma',      # Khối u ác tính
  5: 'https://dermnetnz.org/topics/melanocytic-naevus',
  6: 'https://www.ssmhealth.com/cardinal-glennon/services/pediatric-plastic-reconstructive-surgery/hemangiomas#:~:text=Vascular%20lesions%20are%20relatively%20common,Vascular%20Malformations%2C%20and%20Pyogenic%20Granulomas.' # Tổn thương mạch máu ở da
}

def generate_chart(prediction_probs):
    plt.figure(figsize=[50,50])
    plt.pie(prediction_probs)
    plt.legend(list(SKIN_CLASSES.values()), loc='center', bbox_to_anchor=(0.5, -0.1), prop={'size': 80}) 
    chart_path = os.path.join(app.root_path, 'static/data', 'prediction_chart.png')
    plt.savefig(chart_path, bbox_inches='tight')  # bbox_inches='tight' để cắt bớt không gian trống xung quanh biểu đồ
    plt.close()
    K.clear_session() 
    return chart_path
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['upload']
        for filename in os.listdir(os.path.join(app.root_path, 'static/data')):
            file_path = os.path.join(os.path.join(app.root_path, 'static/data'), filename)
            os.remove(file_path)
        os.makedirs(os.path.join(app.root_path, 'static/data'), exist_ok=True)
        path = os.path.join(app.root_path, 'static/data', f.filename)
        f.save(path)
        j_file = open(os.path.join(app.root_path, 'models/modelnew.json'), 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights(os.path.join(app.root_path, 'models/modelnew.h5'))
        img1 = image.load_img(path, target_size=(224,224))
        img1 = np.array(img1)
        img1 = img1.reshape((1,224,224,3))
        img1 = img1/255
        prediction = model.predict(img1)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        link = SKIN_CLASSES_LINK[pred]
        generate_chart(prediction[0]) 

    return render_template('predict.html', title='Success', skinDiseaseLink=link, predictions=disease, acc=accuracy*100, img_file=f.filename)

@app.route('/health')
def health():
    return "Healthy", 200



