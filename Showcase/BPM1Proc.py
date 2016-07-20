import cvlib

img = cvlib.load("BPM1photo.jpg") # change to fetch
lap = cvlib.binaryThreshold(img, threshVal=90) #MAY NEED TO ADJUST
contours = cvlib.findContours(lap)
cvlib.fillContour(lap, contours[0], color=(100,100,255))

m = cvlib.cntInfo(img, contours[0])
cvlib.plotPoints(lap, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(lap, contours[0])
cvlib.plotPoints(img, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(img, contours[0])
cvlib.plotPoint(img, m["max"], color=(100,100,0), radius=5)
cvlib.plotPoint(lap, m["max"], color=(100,100,0), radius=5)
cvlib.plotCentroid(lap, contours[0])
cvlib.plotCentroid(img, contours[0])
print "Object Details: \n"
cvlib.printCntInfo(img, contours[0])
jet = cvlib.applyColorMap(img, "jet")

cvlib.displayImgs([img, lap, jet])
