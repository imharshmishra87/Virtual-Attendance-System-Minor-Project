import warnings
import pickle
import numpy as np
warnings.filterwarnings('ignore')
import os
import cv2 as cv
from insightface.app import FaceAnalysis 

class Model:
    def __init__(self):
        pass

    def emebeddings(self):

        app=FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0,det_size=(640,640))
        dir=r'D:\projects\Minor Project\Virtual-Attendance-System-Minor-Project\data'
        names=[]
        known_face={}
        for name in os.listdir(dir):
            names.append(name)

        for person in names:
            path=os.path.join(dir,person)
            label=names.index(person)

            embeddings=[]
            
            for i in os.listdir(path):
                img_path=os.path.join(path,i)
                img=cv.imread(img_path)

                if img is None:
                    print('Image not found')
                    continue

                faces=app.get(img)
                if len(faces)==0:
                    print('No faces detected')
                    continue

                embedding=faces[0].embedding
                embeddings.append(embedding)

            if len(embeddings)>0:
                mean_embeddings=np.mean(embeddings,axis=0)
                known_face[person]=mean_embeddings

        with open('known_faces.pkl','wb') as f:
            pickle.dump(known_face,f)


em=Model()
em.emebeddings()



            


        