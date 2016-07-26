import cvlib
import cv2
import numpy as np

def nothing(args):
    pass

# Set up GUI
cv2.namedWindow('image', 0)
# create trackbars for X Y XScale YScale
cv2.createTrackbar('X [pel]','image',0,3296,nothing)
cv2.createTrackbar('Y [pel]','image',0,2472,nothing)
cv2.createTrackbar('X Scale [ct/pel]','image',0,1000,nothing)
cv2.createTrackbar('Y Scale [ct/pel]','image',0,1000,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

def toggleSwitch():
    on = cv2.getTrackbarPos(switch, 'image')
    if on == 1:
        cv2.setTrackbarPos(switch, 'image', 0)
    else:
        cv2.setTrackbarPos(switch, 'image', 1)

# mouse callback
def center(event, x, y, flags, param):
    on = cv2.getTrackbarPos(switch, 'image')
    
    if event == cv2.EVENT_LBUTTONDOWN and on == 1:
        cv2.setTrackbarPos('X [pel]', 'image', x)
        cv2.setTrackbarPos('Y [pel]', 'image', y)
        xpos = x
        ypox = y
        xscale = cv2.getTrackbarPos('X Scale [ct/pel]','image')
        yscale = cv2.getTrackbarPos('Y Scale [ct/pel]','image')
        # SEND TO EPICS HERE
        #cv2.circle(tmp,(x,y),3,(255,0,0),-1) # FILLER

def drawCenter(image):
    x = cv2.getTrackbarPos('X [pel]', 'image')
    y = cv2.getTrackbarPos('Y [pel]', 'image')
    image = cv2.line(image, (x-12,y), (x+12,y), (20, 255,100), 5)
    image = cv2.line(image, (x,y-12), (x,y+12), (20, 255,100), 5)
    
    
img = cvlib.load("sample.tif")

cv2.setMouseCallback('image', center)
tmp = img.copy()
while(1):
    tmp = img.copy()
    drawCenter(tmp)
    cv2.imshow('image', tmp)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        toggleSwitch()
    elif k == ord('r'):
        tmp = img.copy()
    elif k == 32:
        x = cv2.getTrackbarPos('X [pel]', 'image')
        y = cv2.getTrackbarPos('Y [pel]', 'image')
        xpos = x
        ypox = y
        xscale = cv2.getTrackbarPos('X Scale [ct/pel]','image')
        yscale = cv2.getTrackbarPos('Y Scale [ct/pel]','image')
        # SEND TO EPICS HERE
        #cv2.circle(tmp,(x,y),3,(255,0,0),-1) # FILLER
    elif k == 27:
        break
    elif k == 82:
        y = cv2.getTrackbarPos('Y [pel]', 'image')
        cv2.setTrackbarPos('Y [pel]', 'image', y-1)
    elif k == 84:
        y = cv2.getTrackbarPos('Y [pel]', 'image')
        cv2.setTrackbarPos('Y [pel]', 'image', y+1)
    elif k == 81:
        x = cv2.getTrackbarPos('X [pel]', 'image')
        cv2.setTrackbarPos('X [pel]', 'image', x-1)
    elif k == 83:
        x = cv2.getTrackbarPos('X [pel]', 'image')
        cv2.setTrackbarPos('X [pel]', 'image', x+1)
        
    
cv2.destroyAllWindows()
        
        
        