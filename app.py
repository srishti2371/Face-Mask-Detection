import sys
import os
import glob
import re


from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
# Define a flask app
app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
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
        
        model = load_model(r'my_h5_model.h5')
        #face_clsfr=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        img = cv2.imread(file_path)
        (h,w)=img.shape[:2]
        blob=cv2.dnn.blobFromImage(img,1.0,(300,300),(104.0,177.0,123.0))
        prototxtPath=os.path.sep.join([r'C:\Users\shali\Desktop\Mask-Detection-and-Recognition\Mask-Detection-and-Recognition\face detector model','deploy.prototxt'])
        weightsPath=os.path.sep.join([r'C:\Users\shali\Desktop\Mask-Detection-and-Recognition\Mask-Detection-and-Recognition\face detector model','res10_300x300_ssd_iter_140000.caffemodel'])
        net=cv2.dnn.readNet(prototxtPath,weightsPath)
        net.setInput(blob)
        detections=net.forward()
        for i in range(0,detections.shape[2]):
            confidence=detections[0,0,i,2]
    
    
            if confidence>0.5:
                #we need the X,Y coordinates
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype('int')
        
                #ensure the bounding boxes fall within the dimensions of the frame
                (startX,startY)=(max(0,startX),max(0,startY))
                (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
        
                #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
                face=img[startY:endY, startX:endX]
                face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face=cv2.resize(face,(224,224))
                face=img_to_array(face)
                face=preprocess_input(face)
                face=np.expand_dims(face,axis=0)
        
                (mask,withoutMask)=model.predict(face)[0]
        
                #determine the class label and color we will use to draw the bounding box and text
                label='Mask' if mask>withoutMask else 'No Mask'
                color=(0,255,0) if label=='Mask' else (0,0,255)
        
                #include the probability in the label
                label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
                #display the label and bounding boxes
                cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
                cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
        filename = 'static\people_photo\savedImage.jpg'
        cv2.imwrite(filename, img)
        return label
    return None

if __name__ == '__main__':
    app.run(debug=False)