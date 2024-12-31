import cv2
import mediapipe as mp
import serial
import time

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
time.sleep(2)  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def detect_fingers(image, hand_landmarks):
    finger_tips = [8, 12, 16, 20]  
    thumb_tip = 4
    finger_states = [0, 0, 0, 0, 0]  

    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        finger_states[0] = 1  

    for idx, tip in enumerate(finger_tips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_states[idx + 1] = 1 

    return finger_states

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(200, 162, 200), thickness=2),  
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)  
            )
            
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            y_max = max([landmark.y for landmark in hand_landmarks.landmark])

            h, w, _ = image.shape
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            box_expansion = 30  
            x_min = max(0, x_min - box_expansion)
            x_max = min(w, x_max + box_expansion)
            y_min = max(0, y_min - box_expansion)
            y_max = min(h, y_max + box_expansion)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (200, 162, 200), 2)

            fingers_state = detect_fingers(image, hand_landmarks)
            arduino.write(bytes(fingers_state)) 
            print(f"Fingers State: {fingers_state}")

            finger_count = sum(fingers_state)  
            cv2.putText(image, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()