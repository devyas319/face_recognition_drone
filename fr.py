import cv2
import numpy as np
import face_recognition


img_modi = face_recognition.load_image_file('mainimg.jpg')
img_modi = cv2.cvtColor(img_modi,cv2.COLOR_BGR2RGB)
#------to find the face location
face = face_recognition.face_locations(img_modi)[0]
#--Converting image into encodings
train_encode = face_recognition.face_encodings(img_modi)[0]
#----- lets test an image
test = face_recognition.load_image_file('2222.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_encode],test_encode))
copy = test.copy()
cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
if(face_recognition.compare_faces([train_encode],test_encode) == [True]):
    cv2.resizeWindow("output", 400, 300)              # Resize window to specified dimensions
    face = face_recognition.face_locations(copy)[0]
    cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
    cv2.imshow('origional',img_modi)
    cv2.imshow('output', copy)
    cv2.waitKey(0)
else:
    cv2.imshow('origional',img_modi)
    cv2.imshow('output', copy)
    cv2.waitKey(0)