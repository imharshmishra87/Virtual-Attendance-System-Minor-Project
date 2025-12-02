import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis
import pickle

with open(r'D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

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

