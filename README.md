drive.mount('/content/drive')
import os
HOME = ('/content/drive/MyDrive/PS')
print(HOME)
!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image
os.chdir("/content/drive/MyDrive/PS")
!pwd
!yolo task=detect mode=train model=yolov8s.pt data=/content/drive/MyDrive/PS/datasets/data.yaml epochs=100 imgsz=800 plots=True
TESTING MODEL
!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
from ultralytics import YOLO
model=YOLO('/content/drive/MyDrive/PS/runs/detect/train5/weights/best.pt')
import cv2
import pandas as pd
import torch
from matplotlib import pyplot as plt
from IPython.display import clear_output
cap = cv2.VideoCapture('/content/drive/MyDrive/PS/cr.mp4')

count = 0

def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = row.astype(int)
            c = class_list[d]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        clear_output(wait=True)
        display_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
Web App
from google.colab import drive
drive.mount('/content/drive')
import os
HOME = ('/content/drive/MyDrive/PS')
print(HOME)
!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
pip install streamlit -q
!pip install streamlit-lottie -q
%%writefile app.py
import streamlit as st
import cv2
import pandas as pd
import tempfile
from PIL import Image
from ultralytics import YOLO
import json
import streamlit_lottie as st_lottie

model = YOLO('/content/drive/MyDrive/PS/runs/detect/train5/weights/best.pt')

class_list = ['accident', 'car', 'truck']
color_dict = {
    'accident': (255, 0, 0),
    'car': (0, 255, 0),
    'truck': (0, 0, 255)
}

def process_frame(frame):
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    boxes = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(boxes).astype("float")

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row.astype(int)
        c = class_list[d]
        color = color_dict[c]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, f'{c}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    return frame

def load_lottiefile(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def main():
    st.title("Car Crash Detection")
    st.markdown(''':violet[Prateek Srivastav]''')
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                progress_bar.progress(count / total_frames)
                if count % 3 == 0:
                    processed_frame = process_frame(frame)
                    st.image(processed_frame, caption=f'Frame {count}', use_column_width=True)

                count += 1

        except Exception as e:
            st.error(f"An error occurred: {e}")
        cap.release()
        tfile.close()

if name == "main":
    main()
!curl ipv4.icanhazip.com 
!streamlit run app.py & npx localtunnel --port 8501
