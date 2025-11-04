import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2 as cv
import warnings
warnings.filterwarnings('ignore')

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

with open(r'Virtual-Attendance-System-Minor-Project\Source\known_faces.pkl','rb') as f:
    known_face=pickle.load(f)

class testing:
    def __init__(self):
        pass

    def facerecognition(self):
        img=cv.imread(r'Virtual-Attendance-System-Minor-Project\data\Testing-Data\IMG-20251104-WA0008.jpg')
        face=app.get(img)

        testing_data=[]
        for i in range(len(face)):
            test_emb=(face[i].embedding)
            testing_data.append(test_emb)

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
            else:
                detectedfaces.append('Unknown')
        return detectedfaces

fr=testing()
print(fr.facerecognition())
        