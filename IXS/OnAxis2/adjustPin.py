import cvlib
import cv2
import numpy as np

SYS = "XF:10IDD-BI"
DEV = "OnAxis-Cam:1"

XMOTOR = "XF:10IDD-OP{Env:1-Ax:X}Mtr"
ZMOTOR = "XF:10IDD-OP{Env:1-Ax:Z}Mtr"
THETAMOTOR = "XF:10IDD-OP{Spec:1-Ax:Th}Mtr" 

YMOTOR = "XF:10IDD-OP{Env:1-Ax:Y}Mtr"

XMTRSCALE = cvlib.caget(SYS+"{"+DEV+"}Xscale")
YMTRSCALE = cvlib.caget(SYS+"{"+DEV+"}Yscale")

XCENTER = cvlib.caget(SYS+"{"+DEV+"}Xcenter")
YCENTER = cvlib.caget(SYS+"{"+DEV+"}Ycenter")
ANGLE = cvlib.caget(THETAMOTOR+".RBV")
    
def nothing(args):
    pass

cv2.namedWindow('image', 0)

img = cvlib.fetchImg(SYS, DEV)
cv2.createTrackbar('X [pel]','image',0, img.shape[1],nothing) 
cv2.createTrackbar('Y [pel]','image',0, img.shape[0],nothing) 
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
        cvlib.caput(SYS+"{"+DEV+"}"+"Xpos", x)
        cvlib.caput(SYS+"{"+DEV+"}"+"Ypos", y)
        diffX = XCENTER-x
        diffY = YCENTER-y
        moveX = diffX*XMTRSCALE*np.cos(ANGLE*np.pi/180.0)
        moveZ = diffX*XMTRSCALE*np.sin(ANGLE*np.pi/180.0)
        moveY = diffY*YMTRSCALE
        cvlib.caput(XMOTOR+".RLV", moveX)
        cvlib.caput(ZMOTOR+".RLV", moveZ)
        cvlib.caput(YMOTOR+".RLV", moveY)
        XCENTER = cvlib.caget(SYS+"{"+DEV+"}Xcenter")
        YCENTER = cvlib.caget(SYS+"{"+DEV+"}Ycenter")
        ANGLE = cvlib.caget(THETAMOTOR+".RBV")
        
        
        

def drawCenter(image):
    x = cv2.getTrackbarPos('X [pel]', 'image')
    y = cv2.getTrackbarPos('Y [pel]', 'image')
    image = cv2.line(image, (x-24,y), (x+24,y), (20, 255,100), 10)
    image = cv2.line(image, (x,y-24), (x,y+24), (20, 255,100), 10)
    

cv2.setMouseCallback('image', center)
tmp = img.copy()
count = 0
while(1):
    tmp = cvlib.fetchImg(SYS, DEV)
    tmp = np.array(tmp, dtype=np.uint8)
    drawCenter(tmp)
    cv2.imshow('image', tmp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        toggleSwitch()
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
        
        
        
