import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv
import warnings
import time
warnings.filterwarnings('ignore')


with open(r'D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

class testing:
    def __init__(self):
        pass

    def facerecognition(self):
        img=cv.imread(r'D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\Source\test-img.jpeg')
        face=app.get(img)
        test_embedding=[]
        for i in face:
            test_embedding.append(i.embedding)
            x,y,w,h = map(int,i.bbox)

            def cosine(a,b):
                return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
            
            '''Cosine function that is mainly used to check the similarity b/w the test emb and the real emb'''
            
            detectedfaces=[]

            for test_emb in test_embedding:
                bestmatch=None
                max_emb=-1

                '''Know_face is a dictionary that mainly consist of a mean embeddings of all the trained people'''
                
                for labels, emb in known_face.items():
                    data=cosine(test_emb,emb)
                    if data>max_emb:
                        max_emb=data
                        bestmatch=labels
                        
                if max_emb>=0.50:
                    detectedfaces.append(bestmatch)
                    color=(0,255,0)
                    name=f"{bestmatch}"

                else:
                    detectedfaces.append('Unknown')
                    name='Unknown'
                    color=(0,0,255)
                
            cv.rectangle(img,(x,y),(w,h),(0,255,0),2)
            cv.putText(img,str(name),(x,y),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,0.8,color,2)
        cv.imshow('image',img)
        cv.waitKey(0)


        
    def videoreco(self):
        dicto={}
        count=0
        faces=[]
        present=[]

        def cosine(a,b):
            return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
        
        def resizedata(frame,size):
            height=int(frame.shape[0]*size)
            width=int(frame.shape[1]*size)
            
            dim=(width,height)
            return cv.resize(frame,dim)
        
        capture=cv.VideoCapture(0)
        interval=2
        last_detection=0
        starttime=time.time()
        lecture=17
       
        while True:
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

            if time.time()-starttime>=lecture:
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
print(fr.videoreco())
# print(fr.facerecognition())