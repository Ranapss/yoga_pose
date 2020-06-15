import numpy as np
import cv2
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.putText(gray,  
                '{}'.format(fps),  
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