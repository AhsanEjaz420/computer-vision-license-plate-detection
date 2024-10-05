import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import tempfile
import time
import os
import shutil
import torch

# YOLO model path
MODEL_PATH = "D:/yolo/New folder/license_plate_detector.pt"

# EasyOCR languages
LANGUAGES = ['en']

# Create a temporary directory to store processed video frames
temp_dir = tempfile.mkdtemp()

# Initialize YOLO and EasyOCR
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
reader = easyocr.Reader(LANGUAGES)

# Title and Description
st.title("License Plate Detection, Storage, and Search")

# Video Upload
video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

# Dictionary to store license plates and frames
plate_data = {}

# Detection settings
confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)

if video_file:
    # Create a temporary file to store video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.flush()

    # Read uploaded video using OpenCV
    video = cv2.VideoCapture(tfile.name)
    
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame)
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score < confidence_threshold:
                continue

            # OCR on the cropped region
            car_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_text = reader.readtext(car_crop, detail=0)

            if license_plate_text:
                plate = license_plate_text[0]
                
                # Save the plate text and corresponding frame
                plate_data[plate] = frame.copy()  # Save the current frame in memory

                # Also save the frame to a file (optional)
                frame_path = os.path.join(temp_dir, f"{plate}.jpg")
                cv2.imwrite(frame_path, frame)

                # Draw bounding box and text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, plate, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Display frame with detection
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Detected License Plate: {plate}")

        frame_count += 1
        time.sleep(0.05)

    # Release video
    video.release()
    tfile.close()
    os.unlink(tfile.name)

    st.success("Video processed and results stored.")

# Search functionality
st.header("Search for License Plate")
search_query = st.text_input("Enter the license plate number to search:")

if search_query:
    # Check if the license plate exists in stored data
    if search_query in plate_data:
        st.success(f"License plate {search_query} found!")
        st.image(cv2.cvtColor(plate_data[search_query], cv2.COLOR_BGR2RGB), caption=f"Frame with {search_query}")
    else:
        st.error(f"License plate {search_query} not found in the stored data.")
