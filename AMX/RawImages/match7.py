#!/usr/bin/env python
"""
Author: William Watson
This program finds the pin and gripper through all 19
photos.
"""

import cvlib

best = cvlib.load("perfect.bmp")
#bestTh = cvlib.binaryThreshold(best, threshVal=100, invert=False)
contours = cvlib.findContours(best, thresh=100)
contours = contours[1]
gripper = cvlib.load("gripper.bmp")
pin = cvlib.load("pin.bmp")
#cvlib.drawContour(best, contours, thickness = 5)
#cvlib.display(best)
#lst = [best]
for i in range(1,7):
	img = cvlib.load("photo0%s.bmp" % i)
	#imgTh = cvlib.binaryThreshold(img, threshVal=100, invert=False)
	cnts = cvlib.findContours(img, thresh=100)
	cx, cy = cvlib.centroid(cnts[0])
	if cy < 700:
		mount = cnts[0]
	else:
		mount = cnts[1]
	#cvlib.drawContour(img, mount, thickness = 5)
	#cvlib.drawContour(img, contours, thickness = 5, color=(255,255,0))
	print "Image " + str(i) + " " + str(cvlib.matchShapes(contours, mount)) + " Sim"
	p = cvlib.templateMatchSingle(img, pin)
	g = cvlib.templateMatchSingle(img, gripper)
	#cvlib.display(img)
	#lst.append(img)

#cvlib.displayImgs(lst)
