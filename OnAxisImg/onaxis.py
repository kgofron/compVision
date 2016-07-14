import cvlib

filename = "sample_0008.tif" #01-10 or sample.tif

img = cvlib.load(filename)
gray = cvlib.grayscale(img)
lap = cvlib.otsu(img, invert=True)

#lap = cvlib.binaryThreshold(img, threshVal=90)

#tmp = img.copy()
#print tmp[1620][1707]
#tmp = cvlib.floodFill(tmp,(1620, 1707), newval=[255,0,0])
#print laplacian
#print tmp[1620][1707]
#can = cvlib.edgeDetect(lap, minVal=50)
lap = cvlib.closing(lap)
lap = cvlib.opening(lap)
contours = cvlib.findContours(lap)
cvlib.drawContour(img, contours[0])
m = cvlib.cntInfo(img, contours[0])
poi = cvlib.poi(img, contours[0])
for key, value in poi.iteritems():
    cvlib.plotPoint(img, value, radius = 10)
print m
cvlib.displayImgs([img, lap])

