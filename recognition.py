from hashlib import new
import face_recognition
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
from bounding_box import bounding_box as bb
import csv
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt

PLOT_CONFUSION_MATRIX = False
DETECT_ON_VIDEO = False

cascPathface = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
dataset_name = "Celebrity"
dataset_folder = "data/"+dataset_name
test_prefix =  dataset_folder+"/test/"
train_prefix= dataset_folder+"/train/"
clf = pickle.loads(open(dataset_folder+'/SVC.object', "rb").read())
test_file = open(dataset_folder+"/test.csv", "r")
test_csv = csv.reader(test_file)

#region Video 
if DETECT_ON_VIDEO:
    video_capture = cv2.VideoCapture("data/videos/Power5.mp4")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Height", height, "width", width)
    
    cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        #frame = np.array(pyautogui.screenshot())[:,:,::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []
        for encoding in encodings:
            name = clf.predict([encoding])[0]     
            names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            bb.add(frame, x, y, x+w, y+h, label=str(name), color="blue")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1)==27:
            break
    video_capture.release()
    cv2.destroyAllWindows()
#endregion

else:
    predicted = ["Unknown"]
    actual = ["Unknown"]
    header = ["Path","Class"]

    cv2.namedWindow("debug", cv2.WINDOW_KEEPRATIO)
    unknown_encoding = [0]*128
    encodeds = [unknown_encoding]
    for row in test_csv:
        if row == header:
            continue

        image = cv2.imread(row[0])
        if image.data==None:
            print("Can not load", row[0])
            continue
        actual.append(row[1])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        face_bboxs = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.3,
                                            minNeighbors=5,
                                            #minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(face_bboxs)==0:
            print("Can not find any faces on", row[0])
            predicted.append("Unknown")
            encodeds.append(unknown_encoding)        
        else:
            encoding = face_recognition.face_encodings(rgb, num_jitters=3, model="cnn")
            if len(encoding) == 0:
                name="Unknown"
                encodeds.append(unknown_encoding)
            else:
                encodeds.append(encoding[0])
                name = clf.predict(encoding)[0]
            predicted.append(name)
            x, y, w, h = face_bboxs[0]
            bb.add(image, x, y, x+w, y+h, label=str(name), color="blue")
        cv2.imshow("debug", image)
        key = cv2.waitKey(1000)
        if key==27:
            break
        elif key==ord("c"):
            file_name = row[0].split("/")[-1]
            cv2.imwrite("debug/"+file_name, image)
    print(classification_report(actual, predicted))
    if PLOT_CONFUSION_MATRIX:
        plot_confusion_matrix(clf, encodeds, actual)
        plt.show()
