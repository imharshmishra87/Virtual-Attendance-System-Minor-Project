import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv
import warnings
import time
warnings.filterwarnings('ignore')

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

with open(r'Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

class testing:
    def __init__(self):
        pass

    def facerecognition(self):
        img=cv.imread(r'Virtual-Attendance-System-Minor-Project\data\Testing-Data\IMG-20251104-WA0005.jpg')
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
        interval=3
        last_detection=0
        capture=cv.VideoCapture(0)
        while True:
            isTrue, frame=capture.read()
            current_time=time.time()

            if current_time-last_detection>=interval:
                face=app.get(frame)
                last_detection=current_time
            
            testing_data=[]
            for i in face:
                testing_data.append(i.embedding)
                x,y,w,h=map(int, i.bbox)

                def cosine(a,b):
                    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

                detectedfaces=[]

                for test_emb in testing_data:
                    bestmatch=None
                    max_emb=-1

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
                        color=(0,0,255)
                        name="Unknown"

                cv.rectangle(frame,(x,y),(w,h),(0,255,0),2)
                cv.putText(frame,str(name),(x,y),cv.FONT_HERSHEY_DUPLEX,0.8,color,2)
            cv.imshow('Image',frame)
                
            if cv.waitKey(40) and 0Xff==ord('d'):
                break
        

fr=testing()
# print(fr.videoreco())
print(fr.facerecognition())