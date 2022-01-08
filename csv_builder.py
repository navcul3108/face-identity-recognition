from os import listdir, rename, makedirs, write
from os.path import join, abspath
import cv2
import csv

dataset_name = "Celebrity"
dataset_folder = "data/"+dataset_name
test_prefix =  dataset_folder+"/test/"
train_prefix= dataset_folder+"/train/"

train_file = open(f"{dataset_folder}/train.csv", "w", newline="")
train_csv = csv.writer(train_file)
test_file = open(f"{dataset_folder}/test.csv", "w",newline="")
test_csv = csv.writer(test_file)

header = ["Path","Class"]
train_csv.writerow(header)
test_csv.writerow(header)

for face_id in listdir(train_prefix):
    print("face id", face_id)
    folder_prefix = train_prefix+face_id+"/"
    for idx, file_name in enumerate(sorted(listdir(train_prefix+face_id))):
        print("Processing", folder_prefix+file_name)
        train_csv.writerow([folder_prefix+file_name, face_id])
        
for face_id in listdir(test_prefix):
    print("face id", face_id)
    folder_prefix = test_prefix+face_id+"/"
    for idx, file_name in enumerate(sorted(listdir(test_prefix+face_id))):
        print("Processing", folder_prefix+file_name)
        test_csv.writerow([folder_prefix+file_name, face_id])

train_file.close()
test_file.close()