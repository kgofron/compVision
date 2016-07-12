import cvlib

filename = "bulb/T201607061.jpg" #1-3
img = cvlib.load(filename)
gray = cvlib.grayscale(img)
#hsv = cvlib.applyJET(img)
lap = cvlib.binaryThreshold(gray, threshVal=60)

contours = cvlib.findContours(lap)

m = cvlib.cntInfo(img, contours[0])
cvlib.plotPoints(lap, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(lap, contours[0])
cvlib.plotPoints(img, cvlib.extremePointsTup(contours[0]), radius=8)
cvlib.drawContour(img, contours[0])
cvlib.plotPoint(img, m["max"], color=(100,100,0), radius=5)
cvlib.plotPoint(lap, m["max"], color=(100,100,0), radius=5)
cvlib.plotCentroid(lap, contours[0])
print m
jet = cvlib.applyColorMap(img, "jet")
tmp = cvlib.drawMatch(img, cvlib.load("Crystals/bulb/bulbPic.jpg"))
cvlib.displayImgs([img, lap, jet, tmp])