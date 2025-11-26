import pickle 
from datetime import date,timedelta,datetime
import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv
import warnings
import time
warnings.filterwarnings('ignore')
import mysql.connector as mc
import pandas as pd


with open(r'D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

class testing:
    def __init__(self):
        pass

    def getdata(self):
        today=date.today()
        dayname=today.strftime('%a')
        conn=mc.connect(database="test",password="harsh1514",host="localhost",user="root")
        '''Connected with my sql server database'''

        query="SELECT start,end FROM timetable WHERE day=%s order by start asc"
        cursor=conn.cursor()

        '''Executing the query'''

        cursor.execute(query,(dayname,))
        return cursor.fetchall()
    

    def videoreco(self):
        dicto={}
        count=0
        faces=[]

        def cosine(a,b):
            return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
        
        def resizedata(frame,size):
            height=int(frame.shape[0]*size)
            width=int(frame.shape[1]*size)
            
            dim=(width,height)
            return cv.resize(frame,dim)
        



        while True:
            rows=self.getdata()
            for row in rows:
                start=datetime.combine(date.today(),datetime.min.time()) + row[0]
                end=datetime.combine(date.today(),datetime.min.time()) + row[1]       
                
                if start>datetime.now():
                    duration=start-datetime.now()
                    du=duration.total_seconds()
                    print("sleep mode right now")
                    capture.release()
                    time.sleep(du)
                    
                else:
                    capture=cv.VideoCapture(0)
                    interval=2
                    last_detection=0
                    timestamp=time.time()
                    isTrue, frame=capture.read()

                    if not isTrue:
                        print("Frame Doesm't Exists")
                        break

                    resizedframe=resizedata(frame,0.75)
                    if timestamp-last_detection>=interval:
                        faces=app.get(resizedframe)
                        count+=1
                        last_detection=timestamp 

                    for i in faces:
                        test_emb=i.embedding
                        max_emb=-1
                        bestmatch=None
                        for labels, emb in known_face.items():
                            data=cosine(test_emb,emb)
                            if data>max_emb:
                                max_emb=data
                                bestmatch=labels
                        
                        if max_emb>0.50:
                            dicto[bestmatch]=dicto.get(bestmatch,0)+1
                        else:
                            dicto['Unknown']=dicto.get("Unknown",0)+1
                    
            if end>datetime.now():
                break
            capture.release()
            attendance=self.takeattendance(dicto,count)
            print(attendance)

    def takeattendance(self,dicto,count=None):
        l1=[]
        threshold=count*0.20
        for labels, values in dicto.items():
            if values>threshold:
                l1.append(labels)
        return l1

fr=testing()
fr.videoreco()