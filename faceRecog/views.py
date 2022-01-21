from django.shortcuts import render, redirect
import cv2
from . import cascade as casc
from PIL import Image
import cv2
import math
import pickle
from .settings import BASE_DIR
from records.models import Records
from django.http import HttpResponse
from rest_framework.response import Response
from django.http.response import StreamingHttpResponse
from django.views.decorators import gzip
import base64
from django.conf import settings

# Create your views here.
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')


def highlightFace(net, img, conf_threshold=0.7):
    frameOpencvDnn=img.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create()
    # loading the training data
    rec.read(BASE_DIR+'/ml/recognizer/trainingData.yml')
    faceProto=BASE_DIR +'/faceRecog/opencv_face_detector.pbtxt'
    faceModel=BASE_DIR +'/faceRecog/opencv_face_detector_uint8.pb'
    ageProto=BASE_DIR +'/faceRecog/age_deploy.prototxt'
    ageModel=BASE_DIR +'/faceRecog/age_net.caffemodel'
    genderProto=BASE_DIR +'/faceRecog/gender_deploy.prototxt'
    genderModel=BASE_DIR +'/faceRecog/gender_net.caffemodel'

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    font = cv2.FONT_HERSHEY_SIMPLEX
    video=cv2.VideoCapture()
    padding=20
    while cv2.waitKey(1)<0 :
        hasFrame,frame=cam.read()
        if not hasFrame:
            cv2.waitKey()
            break
        flip_image=cv2.flip(frame,1)
        resultImg,faceBoxes=highlightFace(faceNet,flip_image)
        if not faceBoxes:
            print("No face detected")
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            dbgender=gender
            # print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            dbage=age
            # print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
        if(cv2.waitKey(1) == ord('q')):
            cam.release()
            cv2.destroyAllWindows()
            break
   

    record= Records.objects.create(gender=gender,ageGroup=age)
    print(record)
    cam.release()
    data={"age":age,"gender":gender}
    cv2.destroyAllWindows()
    return render(request,'face-details.html', {"age":age,"gender":gender} )

def detectImage(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename =  BASE_DIR+'/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    pca_pkl_filename =  BASE_DIR+'/ml/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca
    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)
    # im.show()
    imgPath = BASE_DIR+'/ml/uploadedImages/'+str(userImage)
    print(imgPath)
    ima=im.save(imgPath, 'JPEG')
    print(ima)



    ''' 
    Input Image
    '''
    # try:
    inputImg = casc.facecrop(imgPath)
    # inputImg.show()
    print("test")
    faceProto=BASE_DIR +'/faceRecog/opencv_face_detector.pbtxt'
    faceModel=BASE_DIR +'/faceRecog/opencv_face_detector_uint8.pb'
    ageProto=BASE_DIR +'/faceRecog/age_deploy.prototxt'
    ageModel=BASE_DIR +'/faceRecog/age_net.caffemodel'
    genderProto=BASE_DIR +'/faceRecog/gender_deploy.prototxt'
    genderModel=BASE_DIR +'/faceRecog/gender_net.caffemodel'

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    font = cv2.FONT_HERSHEY_SIMPLEX
    video=cv2.VideoCapture(imgPath)
    print("test1")
    userId = 0
    padding=20
    while cv2.waitKey(1)<0 :
        print("test2")
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break
    
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        print("imhage")
        if not faceBoxes:
            print("No face detected")
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            print("test3")

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
            if(cv2.waitKey(1) == ord('q')):
                
                cv2.destroyAllWindows()
                break      

    record= Records.objects.create(gender=gender,ageGroup=age)   
    print(record)
    
    data={"age":age,"gender":gender}
    cv2.destroyAllWindows()
    return render(request,'face-details.html', {"age":age,"gender":gender} )

    # except :
    #     print("No face detected, or image not recognized")
    #     return redirect('/error_image')


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()

def index1(request):
	return render(request, 'records.html')
def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


