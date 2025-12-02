import pickle 
from datetime import datetime
import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv
import warnings
import time
from database import databasedata
db=databasedata()
warnings.filterwarnings('ignore')

with open(r'Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

class facerecognition():
    def __init__(self):
        pass

    def cosine(self,a,b):
        return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
    
    def resizedata(self,frame,size):
        height=int(frame.shape[0]*size)
        width=int(frame.shape[1]*size)
        
        dim=(width,height)
        return cv.resize(frame,dim)

    def capturevideo(self):
        dicto={}
        count=0
        faces=[]
        capture=cv.VideoCapture(0)
        interval=2

        while True:
            last_detection=0
            active=db.lecturetime()
            
            if active is None:
                print("No lectures")
                time.sleep(1)
                continue
            start,end=active

            while datetime.now()<end:

                isTrue, frame=capture.read()
                if not isTrue:
                    print("Frame Doesn't Exists")
                    break

                resizedframe=self.resizedata(frame,0.75)

                timestamp=time.time()
                if timestamp-last_detection>=interval:
                    faces=app.get(resizedframe)
                    count+=1
                    last_detection=timestamp 

                for i in faces:
                    test_emb=i.embedding
                    max_emb=-1
                    bestmatch=None
                    for labels, emb in known_face.items():
                        data=self.cosine(test_emb,emb)
                        '''checking the cosine similarity b/w test emb and emb'''
                        if data>max_emb:
                            max_emb=data
                            bestmatch=labels
                    
                    if max_emb>0.50:
                        dicto[bestmatch]=dicto.get(bestmatch,0)+1
                    else:
                        dicto['Unknown']=dicto.get("Unknown",0)+1
    
            '''<-------Lecture Ends---------->'''
            attendance=self.takeattendance(dicto,count)
            print(attendance)
            dicto.clear()
            count=0

    def takeattendance(self,dicto,count=None):
        l1=[]
        if count == 0 or count is None:
            return l1
        threshold=count*0.20
        for labels, values in dicto.items():
            if values>threshold:
                l1.append(labels)
        return l1

fr=facerecognition()
fr.capturevideo()