from flask import Flask, render_template, request

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
new_model = tf.keras.models.load_model('D:\\Major Project\\pneumonia\\review 2\\latest.h5')
import numpy as np
import flask
from keras.preprocessing import image




app = Flask(__name__)

@app.route("/")
def about():
    return render_template("frontend.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('image.jpeg')
  # predicting images
      path = 'image.jpeg'
      img = image.load_img(path, target_size=(150, 150, 3))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      images = np.vstack([x])
      classes = new_model.predict(images, batch_size=10)
      print(path)
      result=''
      if(classes[0][0]>classes[0][1]):
              result='The patient is Normal' 
      else:
              result='The patient has Pneumonia'
      return render_template('result.html',result=result)

if __name__ == "__main__":
    app.run()
