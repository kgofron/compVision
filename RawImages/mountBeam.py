"""
BNL NSLS-2 Computer Vision Program for Gripper Mount
Author: William Watson
Date: 2016-6-29

This Program takes an image and Prints any error strings to the console
to warn any user if the Pin or Gripper is not in the correct position.
In addition, it also processes the image to determine if any issues have occured
and prompts the User to any possible problems.
"""

import cvlib

# Check Positon
def posCheck(img, cnt):
	pts = cvlib.extremePointsTup(cnt)
	shape = cvlib.shape(img)
	for pt in pts:
		if pt[0] < 10:
			print "ERROR: Mount Not Aligned: Left Extreme: " + str(pt)
		if abs(pt[0] - shape[0]) < 10:
			print "ERROR: Mount Not Aligned: Right Extreme: " + str(pt)
		if abs(pt[1] - shape[1]) < 10:
			print "ERROR: Mount Not Aligned: Bottom Extreme: " + str(pt)
	cx, cy = cvlib.centroid(cnt)
	if cx < 250:
		print "ERROR: Mount Centroid: Too Left: " + str(cx)
	if cx > 1500:
		print "ERROR: Mount Centroid: Too Right: " + str(cx)
	if cy < 250:
		print "ERROR: Mount Centroid: Too High: " + str(cy)
	if cy > 700:
		print "ERROR: Mount Centroid: Too Low: " + str(cy)

# Find a Vertical Tangent to Centorid
def verticalTang(cnt):
	cx, cy = cvlib.centroid(cnt)
	minPt = (cnt[0][0][0], cnt[0][0][1])
	minDist = abs(minPt[0] - cx)
	for i in cnt:
		apprx = abs(i[0][0] - cx)
		if apprx < minDist:
			minDist = apprx
			minPt = (i[0][0],i[0][1])
	return minPt

# Separates the Contours
def separateCnt(img, cnts):
	cx, cy = cvlib.centroid(cnts[0])
	if cy < 700:
		return (cnts[0], cnts[1])
	else:
		return (cnts[1], cnts[0])

# Discovers any kink points in contour
def kinks(img, cnt):
	prev = cnt[0][0]
	k = []
	kinkDetect = False
	count = 0
	for pt in cnt:
		if pt[0][0] < prev[0] and abs(pt[0][1] - prev[1]) < 1:
			k.append((pt[0][0], pt[0][1]))
			cvlib.plotPoint(img, (pt[0][0], pt[0][1]), radius = 10, color=(255,55,55))
			kinkDetect = True
			count = count + 1
		prev = pt[0]
	if kinkDetect and len(k) > 0:
		print "ERROR: " + str(count) + " Possible Kink Points Detected: Adjust Gripper"
	if len(k) > 0:
		return k
	else:
		return None

# Splits the kinks if found
def splitKink(kinks):
	if kinks is None:
		return None
	prev = kinks[0]
	CNT = []
	kin = []
	count = 0
	for k in kinks:
		if abs(k[0] - prev[0]) < 20:
			kin.append(k)
		else:
			if len(kin) > 1:
				CNT.append(kin)
				kin = []
		if count == len(kinks) - 1:
			if len(kin) > 1:
				CNT.append(kin)
				kin = []
		prev = k
		count = count + 1
	return sorted(CNT, reverse=True, key=len)

# Find mean of list of points
def mean(lst):
	if len(lst) == 0:
		return None
	X = 0
	Y = 0
	for item in lst:
		X += item[0]
		Y += item[1]
	avgX = X / len(lst)
	avgY = Y / len(lst)
	return (avgX, avgY)

# Finds a Vertical tangent to pt
def vertical(cnt, seq, pt):
	if len(seq) == 0:
		return None
	cx, cy = pt
	minPt = seq[0]
	minDist = abs(minPt[0] - cx)
	for item in cnt:
		apprx = abs(item[0][0] - cx)
		if apprx < minDist and abs(item[0][1] - cy) > 10 and item[0][1] < cy:
			minDist = apprx
			minPt = (item[0][0],item[0][1])
	return minPt

# Finds distance of Kinks
def kinkDistance(image, mount):
	k = kinks(image, mount)
	minPts = []
	if k is not None:
		cnt = splitKink(k)
		count = 0
		for seq in cnt:
			if count == 2:
				break;
			avg = mean(seq)
			minPt = vertical(mount, seq, avg)
			minPts.append((avg, minPt))
			cvlib.plotPoint(image, minPt, radius = 10, color=(0,255,0))
			count = count + 1
	d = {}
	cx, cy = cvlib.centroid(mount)
	for item in minPts:
		if item[0][0] < cx:
			d["L"] = cvlib.distance(item[0], item[1])
		else:
			d["R"] = cvlib.distance(item[0], item[1])
	for key, value in d.iteritems():
		print "ERROR: Possible Kink Distance on " + str(key) + ": " + str(value)
	return d

# Finds all Vertical lines in cnt
def verticalLinesFind(image, cnt):
	prev = apprx[0][0]
	verticalLines = []
	for item in apprx:
		m = cvlib.slope(prev, item[0])
		if prev[0] == item[0][0] and prev[1] == item[0][1]:
			continue
		if m is None or (abs(m) > 5 and abs(prev[1] - item[0][1]) > 50):
			verticalLines.append([(prev[0],prev[1]), (item[0][0], item[0][1]), cvlib.distance(prev, item[0])])
			cvlib.drawLine(image, (prev[0],prev[1]), (item[0][0], item[0][1]), thickness = 15)
		prev = item[0]
	d = None
	if len(verticalLines) < 6:
		print "ERROR: Missing Component / Gripper Not Aligned"
	if len(verticalLines) > 6:
		print "ERROR: Possible Kinks Detected"
	if len(verticalLines) == 6:
		means = []
		for item in verticalLines:
			avg = mean([item[0], item[1]])
			means.append(avg)
		last = len(means) - 1
		d = {"T" : cvlib.distance(means[0], means[last]), "C":cvlib.distance(means[1], means[last-1]), "B":cvlib.distance(means[2], means[last-2]) }
	return {"lines" : verticalLines, "CrossDis" : d}

# Outlines the Gripper
def outlineGripper(image, cnt, YThresh):
	pts = []
	prev = cnt[0][0]
	for i in cnt:
		pt = i[0]
		if pt[1] >= YThresh:
			pts.append(pt)
			cvlib.drawLine(image, (prev[0],prev[1]), (pt[0], pt[1]), color = (100,225,0), thickness = 10)
		prev = pt
	return pts

# Determines possible pin distance
def getPinDistance(image, mount, lines, pts):
	if len(lines["lines"]) > 3 :
		avgY = (lines["lines"][2][0][1] + lines["lines"][3][1][1])/2
		gripPts = outlineGripper(image, mount, avgY)
	pt1 = lines["lines"][1][1][1]
	pt2 = pts["B"][1]
	dis = abs(pt1 - pt2)
	if dis > 250:
		print "ERROR: Pin Not Mounted Correctly: Distance: " + str(dis)
	return dis


def checkSim(mount):
        best = cvlib.load("photo14.bmp")
        bestTh = cvlib.binaryThreshold(best, threshVal=100, invert=False)
        contours = cvlib.findContours(bestTh)
        contours = contours[1]
        print "Image Similiarity" + str(i) + " " + str(cvlib.matchShapes(contours, mount))


def matchGame(img):
        img = cvlib.drawMatch(img, cvlib.load("pin.bmp"))
	img = cvlib.drawMatch(img, cvlib.load("gripper.bmp"), color=(255,0,0))
	cvlib.display(img)
        

##################################################################################

# Load Image
image = cvlib.load("photo7.bmp")
img = image.copy()

#Find COntours, Check Position, Separate Contours
contours = cvlib.findContours(img)
mount, base = separateCnt(img, contours)
posCheck(img, mount)

# Find Points of Value
pts = cvlib.extremePoints(mount)
centroid = cvlib.centroid(mount)
botPt = verticalTang(mount)

# Process Image
apprx = cvlib.contourApprox(mount, epsilon=0.0025)
lines = verticalLinesFind(image, apprx)
d = kinkDistance(image, mount)
pinDist = getPinDistance(image, mount, lines, pts)

#Display Work, Optional
#cvlib.drawContour(image, mount)
cvlib.drawLine(image, centroid, botPt)
cvlib.drawContour(image, apprx, color=(255,255,0))
cvlib.drawContour(image, base, color=(255,0,0))
cvlib.plotPoints(image, cvlib.extremePointsTup(mount), radius = 10)
cvlib.plotCentroid(image, mount)
checkSim(mount)
matchGame(image)
cvlib.display(image, "Photo")

# End of File
