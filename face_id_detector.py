import cv2
import numpy as np
import utils

# 训练模型
[X, y]=utils.read_faces("C:\\Users\\Administrator\\PycharmProjects\\face_detector\\face_generator_root")
y=np.asarray(y, dtype=np.int32)
faceModel=cv2.face.createEigenFaceRecognizer()
faceModel.train(np.asarray(X), np.asarray(y))

# 开始识别
current_userName=''
cap=cv2.VideoCapture(0)
success, frame = cap.read()
color = (0,255,0)
classfier=cv2.CascadeClassifier("D:\BaiduYunDownload\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml")
while success:
    success, frame = cap.read()
    size=frame.shape[:2]
    image=np.zeros(size,dtype=np.float16)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(image, image)
    divisor=8
    h, w = size
    minSize=(w//divisor, h//divisor)
    faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x+w, y+h), color)
                face2Detect=image[x:x+w, y: y+h]
                face2Detect=cv2.resize(face2Detect, (200, 200), interpolation=cv2.INTER_LINEAR)
                params=faceModel.predict(face2Detect)
                if(params[1]>4500):
                    userName=utils.getName(params[0])

                    if(current_userName!=userName):
                        utils.play_sound(userName+".wav")

                    cv2.putText(frame, userName, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                    current_userName = userName
                else:
                    cv2.putText(frame, "UNKNOW", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                    current_userName = ''
    cv2.imshow("test", frame)
    key=cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyWindow("test")



