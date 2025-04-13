import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pygame
from threading import Thread
import time
from datetime import datetime

# Classes for eye status
classes = ['Closed', 'Open']

# Load Haar cascade classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

# Initialize video capture, load model, and set initial variables
cap = cv2.VideoCapture(0)
model = load_model("ds_project.h5")
count_closed_eyes = 0
alarm_on = False
alarm_sound_path = "data/wake_up_alarm.mp3"
msg_sound_path = "data/msg.mp3"
welcome_sound_path = "data/welcome.mp3"
status_left_eye = ''
status_right_eye = ''
welcome_played = False

# Function to start playing alarm sound
def start_alarm(sound_path, play_on_loop=True):
    """Play the alarm sound"""
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    if play_on_loop:
        pygame.mixer.music.play(-1)  # Play on loop
    else:
        pygame.mixer.music.play()

# Function to stop alarm sound and play a message
def stop_alarm():
    """Stop the alarm sound"""
    pygame.mixer.music.stop()
    time.sleep(2)
    play_message_once(msg_sound_path, "It is advisable to consider taking a break or resting.")

# Function to play a message sound once
def play_message_once(sound_path, message):
    """Play the message sound once"""
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        cv2.putText(frame, f"{message}", (7, height-30), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
        continue

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape

    # Convert frame to grayscale for better processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Draw background for the text
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)

    # Get current time and display on frame
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"{current_time}", (400, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Drowsiness Detection System", (10, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        # Play welcome message if not played already
        if not welcome_played:
            time.sleep(1)
            welcome_message = "Welcome to Eyes On Safety, I'll be watching out for you while you are driving."
            t = Thread(target=play_message_once, args=(welcome_sound_path, welcome_message))
            t.daemon = True
            welcome_played = True   # Set the flag to True after playing the welcome sound
            t.start()

        # Detect left eye
        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status_left_eye = np.argmax(pred1)
            break

        # Detect right eye
        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status_right_eye = np.argmax(pred2)
            break

        # If the eyes are closed, start counting
        if status_left_eye == 2 and status_right_eye == 2:
            count_closed_eyes += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count_closed_eyes), (10, 48), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
            # if eyes are closed for 4 consecutive frames, start the alarm
            if count_closed_eyes >= 4:
                cv2.putText(frame, "Warning: Drowsiness Detected!", (80, height-20), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    # play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(alarm_sound_path, True))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 48), cv2.FONT_ITALIC, 0.9, (0, 255, 0), 2)
            count_closed_eyes = 0
            if alarm_on:
                alarm_on = False
                # stop the alarm on a new thread
                t = Thread(target=stop_alarm)
                t.daemon = True
                t.start()

    cv2.imshow("EOS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
#stop_alarm()
cap.release()
cv2.destroyAllWindows()
