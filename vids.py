import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model


model = tf.keras.models.load_model('yoga/model/cnn.model')
CATEGORIES = ['bridge','childs','downwarddog','mountain','plank','seatedforwardbend','tree','trianglepose','warrior1','warrior2']


def prepare(img_array):
    IMG_SIZE = 50
    #img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = np.array(img_array)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    predict = prepare(gray)
    prediction = model.predict(predict)
    prediction = list(prediction[0])

    cv2.putText(gray,  
                '{}'.format(CATEGORIES[prediction.index(max(prediction))]),  
                (50, 50),  
                font, 1,  
                (255, 255, 255),  
                2,  
                cv2.LINE_4) 
    
    count += 1
    if count == 100:
        count =0
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print('\n all done \n')