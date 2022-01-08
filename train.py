import face_recognition
import pickle
import cv2
#from os.path import relpath,dirname,abspath
from sklearn.svm import SVC
import csv
 
dataset_name = "Celebrity"
dataset_folder = "data/"+dataset_name
test_prefix =  dataset_folder+"/test/"
train_prefix= dataset_folder+"/train/"

train_file = open(dataset_folder+"/train.csv", "r")
train_csv = csv.reader(train_file)
print("Training data...")
knownEncodings = []
knownNames = []
header = ["Path","Class"]
# loop over the image paths
for row in train_csv:
    if row == header:
        continue

    imagePath = row[0]
    name = row[1]
    image = cv2.imread(imagePath)
    print(imagePath, "loaded")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='cnn')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb,known_face_locations=boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

knownEncodings.append([0]*128)
knownNames.append("Unknown")

clf = SVC().fit(knownEncodings, knownNames)
#use pickle to save data into a file for later use
f = open(dataset_folder+"/SVC.object", "wb")
print("Saved Classifier")
f.write(pickle.dumps(clf))
f.close()