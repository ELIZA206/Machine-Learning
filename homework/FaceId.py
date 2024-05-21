import cv2
import mediapipe as mp
from math import hypot
import numpy as np

# Инициализация Face и hand
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start захвата видео с камеры
cap = cv2.VideoCapture(0)
finger_tips = [mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP, mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP, mpHands.HandLandmark.THUMB_TIP]
thumb_tip = mpHands.HandLandmark.THUMB_TIP

while True:
    # чтение видео по каждому кадру
    _, frame = cap.read()

    # перевернуть
    frame = cv2.flip(frame, 1)

    # конвертировать BGR image в RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка изображения
    Process = hands.process(frameRGB)
    faces = face_cascade.detectMultiScale(frameRGB, 1.3, 5)
    landmarkList = []

    # Руки в кадре
    if Process.multi_hand_landmarks:
        # обнаружение отпечатков рук
        for handlm in Process.multi_hand_landmarks:
            # подсчет пальцев
            fingers = []
            for tip in finger_tips:
                x, y = int(handlm.landmark[tip].x * frame.shape[1]), int(handlm.landmark[tip].y * frame.shape[0])
                cv2.circle(frame, (x, y), 15, (255, 0, 0), cv2.FILLED)
                if tip == mpHands.HandLandmark.INDEX_FINGER_TIP:
                    if handlm.landmark[tip].y < handlm.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                elif tip == mpHands.HandLandmark.MIDDLE_FINGER_TIP:
                    if handlm.landmark[tip].y < handlm.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                elif tip == mpHands.HandLandmark.RING_FINGER_TIP:
                    if handlm.landmark[tip].y < handlm.landmark[mpHands.HandLandmark.RING_FINGER_DIP].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                elif tip == mpHands.HandLandmark.PINKY_TIP:
                    if handlm.landmark[tip].y < handlm.landmark[mpHands.HandLandmark.PINKY_DIP].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if handlm.landmark[tip].x < handlm.landmark[thumb_tip].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                total_fingers = fingers.count(1)
            print(total_fingers)
            # показ сколько пальцев показано
            cv2.putText(frame, str(total_fingers), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

            # Отрисовка прямоугольника вокруг лица и имя с фамилией
            for (a, b, w, h) in faces:
                cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 255, 0), 2)
                if total_fingers == 2:
                    cv2.putText(frame, 'Sergeev Andrei', (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                elif total_fingers == 1:
                    cv2.putText(frame, 'Andrei', (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "неизвестный", (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            for _id, landmarks in enumerate(handlm.landmark):
                # размеры изображения
                height, width, color_channels = frame.shape

                # x,y координат
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

                # прорисовка рук
            Draw.draw_landmarks(frame, handlm,
                                mpHands.HAND_CONNECTIONS)
    # Display Video and when 'q' is entered, destroy the window
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break