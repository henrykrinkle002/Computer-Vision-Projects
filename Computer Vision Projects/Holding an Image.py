from curses import raw
from operator import index
from pickle import TRUE
from tkinter import image_types
from tracemalloc import start
import mediapipe as mp
import cv2
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
img1 = cv2.imread("zoom.jpg")
img1= cv2.resize(img1, (260, 260))
scale = 0
start_dist = 0
new_size = img1.shape[0]
raw_scale = 0
NON_REACTIVE_ZONE = 5
holding = False
img_x, img_y = 10, 20   # initial image position
grab_offset_x = 0
grab_offset_y = 0

def fingers_closed(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []
    
    if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
        fingers.append(hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y)

    for tip in tips:
        #tip_y and joint_y
        fingers.append(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y)
        print(fingers)
    return fingers


while True:
    success, img = cap.read()
   
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    h1, w1, _ = img1.shape
    # Calculate slice coordinates
    x_start = max(int(img_x), 0)
    y_start = max(int(img_y), 0)
    x_end = min(int(img_x + new_size), img.shape[1])
    y_end = min(int(img_y + new_size), img.shape[0])

    img1_x_start = max(0, -int(img_x))
    img1_y_start = max(0, -int(img_y))
    img1_x_end = img1_x_start + (x_end - x_start)
    img1_y_end = img1_y_start + (y_end - y_start)

    # Only copy if width and height > 0
  
    img[y_start:y_end, x_start:x_end] = img1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]


    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            h, w, _ = img.shape
            (index_x, index_y) = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
            (thumb_x, thumb_y) = (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h))
            (little_x, little_y) = (int(hand_landmarks.landmark[20].x * w), int(hand_landmarks.landmark[20].y * h))
            (ring_x, ring_y) = (int(hand_landmarks.landmark[16].x * w), int(hand_landmarks.landmark[16].y * h))

            cv2.circle(img, (index_x, index_y), 10, (255, 0, 120), cv2.FILLED)
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 120), cv2.FILLED)

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            mid_x = int((index_x + little_x) / 2)
            mid_y = int((index_y + little_y) / 2)

            inside = (
    img_x <= little_x <= img_x + new_size and
    img_y <= little_y <= img_y + new_size and
    img_x <= thumb_x <= img_x + new_size and
    img_y <= thumb_y <= img_y + new_size
                    )

            if (inside) and all(fingers_closed(hand_landmarks)) and not holding:
                holding = True # prepare to hold the image
                grab_offset_x = img_x - mid_x
                grab_offset_y = img_y - mid_y

            if  holding: # holding the image
                img_x = mid_x + grab_offset_x
                img_y = mid_y + grab_offset_y
            
            # was holding but now prepare to release
            if holding and not all(fingers_closed(hand_landmarks)):
                holding = False
                        

        img1 = cv2.resize(img1, (new_size, new_size)) 
        # img[img_y:img_y+new_size, img_x:img_x+new_size] = img1_needed 

    cv2.imshow("Image", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()
