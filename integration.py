from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
import sys
import time
import RPi.GPIO as GPIO
from RPi.GPIO import setmode 

global font, size, fx, fy, fh
global takingData, dataColor
global className, count
global showMask

#GPIO.cleanup()

BL=26
FL=24
ENA=22
BR=8
FR=10
ENB=12
sleeptime=1

GPIO.setmode(GPIO.BOARD)
GPIO.setup(FR, GPIO.OUT)
GPIO.setup(BR, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(FL, GPIO.OUT)
GPIO.setup(BL, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

GPIO.output(ENA, GPIO.LOW)
GPIO.output(FR, GPIO.LOW)
GPIO.output(BR, GPIO.LOW)
GPIO.output(ENB, GPIO.LOW)
GPIO.output(FL, GPIO.LOW)
GPIO.output(BL, GPIO.LOW) 


dataColor = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
className = 'NONE'
count = 0
showMask = 0


classes = 'NONE ONE TWO FIVE'.split()


def stop(x):
 GPIO.output(ENA, GPIO.LOW)    
 GPIO.output(BR, GPIO.LOW)
 GPIO.output(FR, GPIO.LOW)
 GPIO.output(ENB, GPIO.LOW)    
 GPIO.output(BL, GPIO.LOW)
 GPIO.output(FL, GPIO.LOW)
 print("Stop")
 time.sleep(x)
 GPIO.output(BR, GPIO.LOW)
 GPIO.output(BL, GPIO.LOW)

def forward(x):
 GPIO.output(ENA, GPIO.HIGH)    
 GPIO.output(FR, GPIO.HIGH)
 GPIO.output(BR, GPIO.LOW)
 GPIO.output(ENB, GPIO.HIGH)    
 GPIO.output(FL, GPIO.HIGH)
 GPIO.output(BL, GPIO.LOW)
 print("Moving Forward")
 time.sleep(x)
 GPIO.output(FR, GPIO.LOW)
 GPIO.output(FL, GPIO.LOW)


def Fright(x):
 GPIO.output(ENA, GPIO.HIGH)    
 GPIO.output(FR, GPIO.HIGH)
 GPIO.output(BR, GPIO.LOW)
 GPIO.output(ENB, GPIO.HIGH)    
 GPIO.output(FL, GPIO.LOW)
 GPIO.output(BL, GPIO.LOW)
 print("Moving Forward Right")
 time.sleep(x)
 GPIO.output(FR, GPIO.LOW)
 GPIO.output(FL, GPIO.LOW)

def Fleft(x):
 GPIO.output(ENA, GPIO.HIGH)    
 GPIO.output(FR, GPIO.LOW)
 GPIO.output(BR, GPIO.LOW)
 GPIO.output(ENB, GPIO.HIGH)    
 GPIO.output(FL, GPIO.HIGH)
 GPIO.output(BL, GPIO.LOW)
 print("Moving Forward Left")
 time.sleep(x)
 GPIO.output(FR, GPIO.LOW)
 GPIO.output(FL, GPIO.LOW)


def initClass(name):
    global className, count
    className = name
    os.system('mkdir -p data/%s' % name)
    count = len(os.listdir('data/%s' % name))


def binaryMask(img):
    kernel = np.ones((3,3),np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


model = load_model('hand_model_gray_second.hdf5')

x0, y0, width = 200, 40, 300

cam = cv2.VideoCapture(0)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('t1.avi', fourcc, 20.0, (640, 480))

while True:
    # Get camera frame
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # mirror
    window = copy.deepcopy(frame)
    cv2.rectangle(window, (x0,y0), (x0+width-1,y0+width-1), dataColor, 12)

    # draw text
    if takingData:
        dataColor = (0,250,0)
        cv2.putText(window, 'Data Taking: ON', (fx,fy), font, 1.2, dataColor, 2, 1)
    else:
        dataColor = (0,0,250)
        cv2.putText(window, 'Data Taking: OFF', (fx,fy), font, 1.2, dataColor, 2, 1)
    cv2.putText(window, 'Class Name: %s (%d)' % (className, count), (fx,fy+fh), font, 1.0, (245,210,65), 2, 1)

    # get region of interest
    roi = frame[y0:y0+width,x0:x0+width]
    roi = binaryMask(roi)

    # apply processed roi in frame
    if showMask:
        window[y0:y0+width,x0:x0+width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # take data or apply predictions on ROI
    if takingData:
         cv2.imwrite('data/{0}/{0}_{1}.png'.format(className, count), roi)
         count += 1
    else:
        img = np.float32(roi)/255.
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        pred = classes[np.argmax(model.predict(img)[0])]
        cv2.putText(window, 'Prediction: %s' % (pred), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
        # use below for demoing purposes
        #cv2.putText(window, 'Prediction: %s' % (pred), (x0,y0-25), font, 1.0, (255,0,0), 2, 1)

    # show the window
    cv2.imshow('Original', window)
    # Keyboard inputs
    key = cv2.waitKey(10) & 0xff

    if (pred == "ONE"):
        forward(5)
    elif (MINPUT == "TWO"):
        Fright(5)
    elif (MINPUT == "FIVE"):
        Fleft(5)
    else:
        stop(0)


    # use q key to close the program
    if key == ord('q'):
        break

    # adjust the size of window
    #elif key == ord('z'):
    #    width = width - 5
    #elif key == ord('a'):
    #    width = width + 5

cam.release()
cv2.destroyAllWindows()

