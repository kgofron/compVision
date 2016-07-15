#import cvlib
from matplotlib import pyplot as plt
import cv2
#img = cvlib.fetchImg("XF:10IDD-BI", "OnAxis-Cam:1")

img = cv2.imread("T20160715.jpg")
print img
<<<<<<< HEAD
print len(img)

=======
#cvlib.display(img)
>>>>>>> 89f7c471920554aa5ee8c273202f633acc8f133b
#cvlib.save(img)
"""
lap = cvlib.binaryThreshold(img, threshVal=100)
contours = cvlib.findContours(lap)
cvlib.display(lap)

cvlib.fillContour(lap, contours[0], color=(100,100,255)) 

m = cvlib.cntInfo(img, contours[0])
cvlib.plotPoints(lap, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(lap, contours[0])
cvlib.plotPoints(img, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(img, contours[0])
cvlib.plotPoint(img, m["max"], color=(100,100,0), radius=5)
cvlib.plotPoint(lap, m["max"], color=(100,100,0), radius=5)
cvlib.plotCentroid(lap, contours[0])
print "Object Details: " + str(m)

jet = cvlib.applyColorMap(img, "jet")
"""
"""
f1 = cvlib.highPassFilter(img)
f2 = cvlib.lowPassFilter(img)
enh = cvlib.enhance(img, window=60)
sharp = cvlib.sharpen(img)

#cvlib.matplotlibDisplay(f1, title="Mag Spoec")
#cvlib.matplotlibDisplay(f2, title="Mag Spoec")
cvlib.matplotlibDisplayMulti([f1, f2, enh, cvlib.grayscale(img), cvlib.grayscale(sharp)])
#cvlib.displayImgs([img, f1, f2])
"""
