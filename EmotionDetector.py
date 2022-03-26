import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
logo_dict = {"Angry" : "data/emojis/angry (2).png" , "Disgusted" : "data/emojis/disgusted.png" , "Fearful" : "data/emojis/fearful.png" , "Happy" : "data/emojis/Happy.png" , "Neutral" : "data/emojis/neutral.png" ,"Sad" : "data/emojis/sad.png" , "Surprised" : "data/emojis/surprised.png"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

while True:

    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # This just to put the emoji on the vedio
        for i , j in logo_dict.items():
            if emotion_dict[maxindex] == i :
                logo = cv2.imread(j)
                size = 500
                logo = cv2.resize(logo, (size, size))
                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                roi = frame[-size-10:-10, -size-10:-10]
                roi[np.where(mask)] = 0
                roi += logo
    
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
