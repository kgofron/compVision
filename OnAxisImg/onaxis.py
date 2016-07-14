import cvlib

filename = "sample_0008.tif" #01-10 or sample.tif

img = cvlib.load(filename)
gray = cvlib.grayscale(img)
lap = cvlib.otsu(img, invert=True)
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

