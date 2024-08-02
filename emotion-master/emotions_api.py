import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from flask import Flask, request

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "sad", 5: "surprise", 6: "neutral"}

app = Flask(__name__)

@app.route('/emotion', methods=['POST'])
def emotion():
    image_upload = request.files["image"]
    image_upload.save(image_upload.filename)

    img = cv2.imread(image_upload.filename)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(img_gray,scaleFactor=1.3, minNeighbors=5)
    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotions.append(emotion_dict[maxindex])

    return str(emotions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)