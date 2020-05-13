from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt 
import os
import cv2
import random
import pickle
from tqdm import tqdm
pose = Image.open('training_set//bridge//File1.jpg').convert('LA')

categories = ['bridge','childs','downwarddog','mountain','plank','seatedforwardbend','tree','trianglepose','warrior1','warrior2']
datadir = 'training_set'
im_size = 50

training_data = []

for category in tqdm(categories):
    path = os.path.join(datadir,category)
    class_num = categories.index(category) 
    for im in os.listdir(path):
        try:
            im_arr = Image.open(os.path.join(path,im)).convert('L')
            new_im_arr = im_arr.resize((im_size,im_size),Image.ANTIALIAS)
            #print(type(new_im_arr))
            training_data.append([np.array(new_im_arr),class_num])
        except:
            pass
random.shuffle(training_data)

X = [] #features 
y = [] # labels 


for feat, label in training_data:
    X.append(feat)
    y.append(label)

print(len(X))
print(len(y))

X = np.array(X).reshape(-1, im_size, im_size, 1)
print(X[1].shape)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
