import pandas as pd
from sklearn.decomposition import PCA
import csv
import cv2
import face_recognition
import matplotlib.pyplot as plt

dataset_name = "Celebrity"
dataset_folder = "data/"+dataset_name
train_file = open(dataset_folder+"/train.csv", "r")
train_csv = csv.reader(train_file)

known_encodings = []
known_names = []

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
        known_encodings.append(encoding)
        known_names.append(name)

fig = plt.figure()
ax = plt.axes(projection='3d')
pca = PCA(n_components=3)
new_data = pca.fit_transform(known_encodings)
x,y,z = zip(*new_data)
data = pd.DataFrame({"x":x,"y":y, "z":z,"name":known_names})
groups = data.groupby("name")
for name, group in groups:
    ax.plot3D(group["x"], group["y"], group["z"], marker="o", linestyle="", label=name)
plt.legend()
plt.show()