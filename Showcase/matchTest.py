#!/usr/bin/env python
"""
Author: William Watson
This program finds the pin and gripper through all 19
photos.
"""

import cvlib

best = cvlib.load("perfect.bmp")
bestTh = cvlib.binaryThreshold(best, threshVal=100, invert=False)
contours = cvlib.findContours(bestTh)
contours = contours[1]
cvlib.drawContour(best, contours, thickness = 5)
#cvlib.display(best)
lst = [best]
for i in range(1,5):
	img = cvlib.load("photo0%s.bmp" % i)
	imgTh = cvlib.binaryThreshold(img, threshVal=100, invert=False)
	cnts = cvlib.findContours(imgTh)
	cx, cy = cvlib.centroid(cnts[0])
	if cy < 700:
		mount = cnts[0]
	else:
		mount = cnts[1]
	cvlib.drawContour(img, mount, thickness = 5)
	#cvlib.drawContour(img, contours, thickness = 5, color=(255,255,0))
	print "Image " + str(i) + " " + str(cvlib.matchShapes(contours, mount)) + " Sim"
	img = cvlib.drawMatch(img, cvlib.load("pin.bmp"))
	img = cvlib.drawMatch(img, cvlib.load("gripper.bmp"), color=(255,0,0))
	#cvlib.display(img)
	lst.append(img)

cvlib.displayImgs(lst)
