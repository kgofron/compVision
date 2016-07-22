import cvlib

img = cvlib.load("BPM2-0.jpg") # change to fetch
lap = cvlib.binaryThreshold(img, threshVal=253)
contours = cvlib.findContours(lap)
cvlib.fillContour(lap, contours[0], color=(100,100,255))

m = cvlib.cntInfo(img, contours[0])
cvlib.plotPoints(lap, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(lap, contours[0])
cvlib.plotPoints(img, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(img, contours[0])
cvlib.plotPoint(img, m["max"], color=(100,100,0), radius=8)
cvlib.plotPoint(lap, m["max"], color=(100,100,0), radius=8)
cvlib.plotCentroid(lap, contours[0], radius=10)
cvlib.plotCentroid(img, contours[0], radius=10)
print "Object Details: " + str(m)
jet = cvlib.applyColorMap(img, "jet")

cvlib.displayImgs([img, lap, jet])
