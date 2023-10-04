import matplotlib.pyplot as plt
import math
import time
from PIL import Image, ImageTk
import cvzone
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2

def upload_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    cap = cv2.VideoCapture(filepath)
    model = YOLO("../Yolo-Weights/yolov8n.pt")


    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break 
    cap.release()
    cv2.destroyAllWindows()
    # Perform necessary operations with the video file

def upload_image():

    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    model = YOLO('../Yolo-Weights/yolov8l.pt')
    results = model(filepath, show=True)
    cv2.waitKey(0)
    # Perform necessary operations with the image file

def open_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    model = YOLO("../Yolo-Weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

# Create the main window
window = tk.Tk()
window.title("Real Time Object Detector")
heading_label = tk.Label(window, text="Object Detection", font=("Helvetica", 24, "bold"))
heading_label.pack()
window.geometry("640x600")
canvas_width = 640
canvas_height = 400
# Create the canvas
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()
# Add background image
background_image = Image.open("C:/Users/vanap/PycharmProjects/ObjectDetection/detection.png")
background_image = background_image.resize((canvas_width, canvas_height), Image.ANTIALIAS)
background_photo = ImageTk.PhotoImage(background_image)
canvas.create_image(0, 0, image=background_photo, anchor=tk.NW)
# Create buttons for each option
video_button = tk.Button(window, text="Upload Video", command=upload_video,bg="sky blue")
video_button.pack(pady=10)

image_button = tk.Button(window, text="Upload Image", command=upload_image,bg="light green")
image_button.pack(pady=10)

webcam_button = tk.Button(window, text="Open Webcam", command=open_webcam,bg="yellow")
webcam_button.pack(pady=10)
window.mainloop()

