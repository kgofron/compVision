"""
Author: William Watson
This program processes images from Scintillator x-ray streaks
Images by Kaz Gofron, NSLS-2
"""

import cvlib

img = cvlib.load("sin1.jpg") #sin1 or sin2
lap = cvlib.binaryThreshold(img, threshVal=20)
contours = cvlib.findContours(lap)
cvlib.fillContour(lap, contours[0], color=(100,100,255)) ##############################

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

cvlib.displayImgs([img, lap, jet])