from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import pandas as pd
import cv2 
import math
from PIL import Image
import os
app=Flask(__name__)

model_path='RFmodel_0.8.pkl'
with open(model_path,'rb') as file:
    model=pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')


ALLOWED_EXTENSIONS=['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

file_name=""
@app.route('/upload' , methods=["POST"])
def upload():
    if 'videofile' not in request.files:
        return "No video file found"
    video=request.files['videofile']
    if video.filename=='':
        return "No video file selected"
    if video and allowed_file(video.filename):
        video.save('static/videos/'+video.filename)
        global file_name
        file_name=video.filename
        return render_template('index.html',uploaded=str(file_name))
    return "invalid filetype"


@app.route('/extract',methods=["POST"])
def extract():
    dir="frames"
    list=os.listdir(dir)

    for i in (list):
        if os.path.exists(dir+'/'+i):
            os.remove(dir+'/'+i)
        else:
            break
    
    video=cv2.VideoCapture('static/videos/'+file_name)
    fps=video.get(cv2.CAP_PROP_FPS)
    fps=math.trunc(fps)

    def isNBI(frame):
        val=0
        flag=0
        for i in range(1226,1252):
            for j in range(37,48):
                for k in range(0,3):

                    val+=frame[j][i][k]
                    if(val>0):
                        return 1
        return 0
    
    n=0
    i=0
    path="frames"
    while True:
        ret,frame=video.read()
        if(n)%fps==0:
            if(isNBI(frame)==1):
                cv2.imwrite("frames//%d.jpg"%i,frame)
                i+=1
        n+=1
        if ret==False:
            break
    video.release()
    cv2.destroyAllWindows()

    dir="frames"
    save_dir="frames"
    list=os.listdir(dir)

    count=i
    #count=n
    for i in list:
        open_name=dir+'/'+i
        img=Image.open(open_name)
        imgCrop=img.crop(box=(360,0,1280,720))
        save_name=save_dir+'/'+i
        imgCrop.save(save_name)

    return render_template("index.html",frames=str(count))

@app.route('/predict', methods=["POST"])
def predict():
    dir="frames"
    list=os.listdir(dir)

    if len(list)==0:
        return "Not data found"

    count=0
    pred=[0,0]
    SIZE=128

    def feature_extractor(dataset):
        x_train=dataset
        image_dataset=pd.DataFrame()
        for image in range(x_train.shape[0]):
            
            df=pd.DataFrame()
            input_image=x_train[image,:,:,:]
            img=input_image
            
            pixel_values=img.reshape(-1)  ##Feature 1 :: the pixel values
            df['Pixel_Value']=pixel_values
            
            num=1
            kernels=[]
            
            for theta in range(3):  ##Feature 2 :: feature from gabor filters
                theta=theta/4. * np.pi
                for sigma in (1,3,5):
                    lamda=np.pi/4
                    gamma=0.5
                    gabor_label="Gabor"+str(num)
                    ksize=9
                    kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    #rint(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
            image_dataset=pd.concat([image_dataset, df],ignore_index=True)
        return image_dataset

    for i in list:

        count+=1
        img_path=dir+'/'+i
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.resize(img,(SIZE,SIZE))
        img=img/255
        input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
        input_img_features=feature_extractor(input_img)
        input_img_features = np.expand_dims(input_img_features, axis=0)
        input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
        img_prediction = model.predict(input_img_for_RF)
        classification=img_prediction[0]

        if(classification==0):
            pred[0]+=1
        else:
            pred[1]+=1

    positive=pred[1]/count
    negative=pred[0]/count


    return render_template("index.html", Positive_prc=str(positive*100),Negative_prc=str(negative*100), Positive_frame=str(pred[1]),Negative_frame=str(pred[0]))






if __name__=='__main__':
    app.run(debug=True)