#!pip install opencv-python !pip install tensorflow !pip install keras !pip install Pillow --dowload

import cv2
import tensorflow
import keras 
from PIL import Image

face_cascade = "haarcascade_frontalface_default.xml"

webcam = cv2.VideoCapture(1)
success, image_bgr = webcam.read()

image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])



#ใส่ภาพที่ต้องการให้AIเรียนรู้
count = 9999
while True:
    success, image_bgr = webcam.read()

    image_org = image_bgr.copy()
    
    image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(image_bw)

    print(f'There are {len(faces)} faces found.')

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('non-mask/non_mask_{count}.jpg',image_org[y:y+h,x:x+w])

    cv2.imshow("Faces found", image_bgr)
    cv2.waitKey(1) 
    
    while True:
        success, image_bgr = webcam.read()
        image_org = image_bgr.copy()
        image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

        faces = face_classifier.detectMultiScale(image_bw)

        for face in faces:
            x, y, w, h = face
            
            cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            image = cface_rgb


            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            data[0] = normalized_image_array

            prediction = model.predict(data)
            
            print(prediction)
            
            if prediction[0][0] > prediction[0][1]:
                cv2.putText(image_bgr,'Masked',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(image_bgr,'Non-Masked',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow("Mask Detection", image_bgr)
        cv2.waitKey(1) 
