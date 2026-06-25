import cv2
import time
import torch
import pandas as pd

from detector import SeniorCitizenDetector
from utils import CSVLogger


# ----------------------------------------------------
# Initialize Detector
# ----------------------------------------------------

detector = SeniorCitizenDetector()

logger = CSVLogger("visits.csv")


# ----------------------------------------------------
# GPU Information
# ----------------------------------------------------

if torch.cuda.is_available():

    gpu_name = torch.cuda.get_device_name(0)

else:

    gpu_name = "CPU"


print("="*60)
print("Senior Citizen Identification System")
print("="*60)
print("Device :", gpu_name)
print("="*60)


# ----------------------------------------------------
# Camera
# ----------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)


if not cap.isOpened():

    print("Cannot Open Camera")

    exit()


# ----------------------------------------------------
# Variables
# ----------------------------------------------------

previous_time = time.time()

fps = 0

total_persons = 0

senior_count = 0

male_count = 0

female_count = 0

font = cv2.FONT_HERSHEY_SIMPLEX

print("\nPress Q to Exit\n")


# ----------------------------------------------------
# Main Loop
# ----------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame = cv2.flip(frame,1)

    frame, detections = detector.detect(frame)

    total_persons = len(detections)

    senior_count = 0

    male_count = 0

    female_count = 0
    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    for det in detections:

        if det["gender"] == "Male":

            male_count += 1

        else:

            female_count += 1

        if det["status"] == "Senior Citizen":

            senior_count += 1

            logger.log(

                det["age"],

                det["gender"],

                det["status"]

            )


    # -----------------------------------------
    # FPS
    # -----------------------------------------

    current = time.time()

    fps = 1 / (current - previous_time)

    previous_time = current


    # -----------------------------------------
    # Header
    # -----------------------------------------

    cv2.rectangle(

        frame,

        (0,0),

        (640,90),

        (40,40,40),

        -1

    )
    cv2.putText(

        frame,

        "Senior Citizen Identification",

        (10,25),

        font,

        0.7,

        (255,255,255),

        2

    )


    cv2.putText(

        frame,

        f"Device : {gpu_name}",

        (10,50),

        font,

        0.55,

        (0,255,255),

        2

    )


    cv2.putText(

        frame,

        f"FPS : {int(fps)}",

        (500,25),

        font,

        0.7,

        (0,255,0),

        2

    )
    # -----------------------------------------
    # Statistics Panel
    # -----------------------------------------

    cv2.rectangle(

        frame,

        (0,390),

        (640,480),

        (40,40,40),

        -1

    )


    cv2.putText(

        frame,

        f"Persons : {total_persons}",

        (10,420),

        font,

        0.65,

        (255,255,255),

        2

    )


    cv2.putText(

        frame,

        f"Senior : {senior_count}",

        (180,420),

        font,

        0.65,

        (0,0,255),

        2

    )


    cv2.putText(

        frame,

        f"Male : {male_count}",

        (340,420),

        font,

        0.65,

        (255,0,0),

        2

    )


    cv2.putText(

        frame,

        f"Female : {female_count}",

        (470,420),

        font,

        0.65,

        (255,0,255),

        2

    )
    cv2.imshow(

        "Senior Citizen Identification System",

        frame

    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):

        break


# ----------------------------------------------------
# Cleanup
# ----------------------------------------------------

cap.release()

cv2.destroyAllWindows()


print("\nApplication Closed")

print("CSV Saved Successfully")                