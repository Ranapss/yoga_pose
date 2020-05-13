from PIL import Image
import numpy as np 


pose = Image.open('training_set//bridge//File1.jpg')
red,blue,green = pose.split()
pose_arr = np.asarray(pose)

print(pose_arr.shape)

pose_arr.show()