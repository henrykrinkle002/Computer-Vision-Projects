from operator import index
from unittest import result
import cv2
import mediapipe as mp
import math


class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), 2)
        cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 55),cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    def CheckClick(self, img, indexFinger):
        x, y = self.pos
        index_Finger_x, index_Finger_y = indexFinger
        
        if (x < index_Finger_x <= x + self.width) & (y <= index_Finger_y < y + self.height):
                cv2.rectangle(img, self.pos, (x + self.width, y + self.height), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, self.value, (x + 40, y + 55), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
                return True
        return False
        

myEquation = "10+5"
distance = 1000
delayCounter = 0
cap = cv2.VideoCapture(0)
#screen resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 420)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.7)

mpDraw = mp.solutions.drawing_utils

buttonListValue = ['7', '8', '9', '/',
                   '4', '5', '6', '*',
                   '1', '2', '3', '-',
                   '0', '.' ,'=', '+']


buttonList = []

cell_size = (370//4)
for y in range(4):
    for x in range(4):
        xpos = x * cell_size + 24
        ypos = y * cell_size + 130
        buttonList.append(Button((xpos, ypos), 85, 85, buttonListValue[y*4+x]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BLUE,  GREEN, RED -> RED,  GREEN, BLUE
    result = hands.process(imgRGB)  #detect hands OPENCV reads video in BGR format but MEDIAPIPE expects RGB

    cv2.rectangle(img, (10, 20), (400, 500), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(img, (20, 27), (390, 490), (0, 0, 0), 3)

    for button in buttonList:
        button.draw(img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #1 hand 
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS) #connections = 21
            h, w, c = img.shape

            (index_x, index_y) = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h) #converting normalized landmark to pixel
            (thumb_x, thumb_y) = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            cv2.circle(img, (index_x, index_y), 10, (255, 0, 120), cv2.FILLED)
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 120), cv2.FILLED)

    if(distance < 50) and (delayCounter == 0):
        for button in buttonList:
            if(button.CheckClick(img, (index_x, index_y))):
                # if delayCounter > 100:
                #     delayCounter = 0
                if (button.value == '='):  
                    try:      
                        myEquation = str(eval(myEquation))
                        print(myEquation)
                    except:
                        myEquation = ""
                else:
                    myEquation += button.value
                delayCounter = 1 # I clicked

    # Avoid duplicates outside the click condition  
    # The Delay occurs here after the loop before clicking 
    if delayCounter !=  0: #if delayCounter is running
        delayCounter += 1 # counting frames till last click
        if delayCounter > 10: # but after 10 frames reset delayCounter
            delayCounter = 0 # ready to click again


    cv2.putText(img, myEquation, (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
         
    cv2.imshow("Image", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break; 

cap.release()
cv2.destroyAllWindows()