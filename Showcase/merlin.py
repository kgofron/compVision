"""
Author: William Watson
Program processes Merlin Streaks
Images Provided by IXS Staff at NSLS-2. Yong Cai, Alessandro Cunsolo, Alexy Suvorov
"""

import cvlib

img = cvlib.load("merlin.jpg")
gray = cvlib.grayscale(img)
lap = cvlib.binaryThreshold(img, threshVal=20)
contours = cvlib.findContours(lap)
contours = cvlib.filterCnts(contours)

a = cvlib.cntInfoMult(img, contours)
crops = [img]
for i in range(0, len(a)):
        print "Object %d:\n" % i
        cvlib.printCntInfo(img, contours[i])
	cvlib.plotPOI(img, contours[i], radius=1)
	crops.append(cvlib.copy(cvlib.cropCnt(img, contours[i])))
        print "\n"
	
cvlib.drawContours(img, contours, thickness = 1)
jet = cvlib.applyColorMap(gray, "jet")

cvlib.displayImgs([crops[0], jet])
