import cv2

# import numpy as np

face_cascade = cv2.CascadeClassifier('face_detector.xml')

test_image = cv2.imread('./images/ceuse_selfie.jpg')

# faces = face_cascade.detectMultiScale(test_image, 1.1, 4)

faces = face_cascade.detectMultiScale(test_image, scaleFactor = 1.2, minNeighbors = 5)

print('Faces found: ', len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0 ,0), 2)

cv2.imwrite('face_detected.png', test_image)
print('Successfully Saved')

