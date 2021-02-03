from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow
import json
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'tomato_cnn.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, models):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = models.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
        monkey_breeds_dict = {"[0]": "Tomato___Bacterial_spot ", 
                      "[1]": "Tomato___Early_blight",
                      "[2]": "Tomato___healthy",
                      "[3]": "Tomato___Late_blight",
                      "[4]": "Tomato___Leaf_Mold ",
                      "[5]": "Tomato___Septoria_leaf_spot",
                      "[6]": "Tomato___Spider_mites Two-spotted_spider_mite",
                      "[7]": "Tomato___Target_Spot",
                      "[8]": "Tomato___Tomato_mosaic_virus",
                      "[9]": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"}
        monkey = monkey_breeds_dict[str(pred)]             # Convert to string
        return monkey
    return None




if __name__ == '__main__':
    app.run(debug=True)