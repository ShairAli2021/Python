import face_recognition
import cv2
import numpy as np
import csv
import os

from datetime import datetime
video_capture= cv2.VideoCapture(0)

kareen_image=face_recognition.load_image_file("./photos/kareen.jpg")
kareen_encoding=face_recognition.face_encodings(kareen_image)[0]


ugala_image=face_recognition.load_image_file('photos/ugala.jpg')
ugala_encoding=face_recognition.face_encodings(ugala_image)[0]

ali_image=face_recognition.load_image_file("photos/ali.jpg")
ali_encoding=face_recognition.face_encodings(ali_image)[0]


know_face_encoding=[
    kareen_encoding,
    ugala_encoding,
    ali_encoding
    
]
know_face_names=[
    "kareen",
    "ugala",
    "ali"
]
students=know_face_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True
now= datetime.now()
current_date= now.strftime("%H-%M-%S")
f=open(current_date+'.csv','w+', newline= "")
Inwrite=csv.writer(f)
while True:
    _,frame= video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings= face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_names= []
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(know_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(know_face_encoding,face_encoding)
        
            best_match_index=np.argmin(face_distance)
          
        
            if matches[best_match_index]:
                
                name=know_face_names[best_match_index]
            face_names.append(name)
            if name in know_face_names:
                if name in students:
                    students.remove(name)
                    position =(10,50)              
                    frame=cv2.putText(frame,"present", position,cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
                    cv2.imshow("attance system",frame)
                    
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    Inwrite.writerow(["name", name , " is present" ,"time  is " , current_time])
                    
                else:
                    frame=cv2.putText(frame,"not present", position,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
                    cv2.imshow("attance system",frame)
                    
                #      print("sorry")
                    #  current_date=now.strftime("%H-%M-%S")
                    #  Inwrite.writerow(["sorry "])
                   
    
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()  


