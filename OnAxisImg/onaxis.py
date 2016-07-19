import cvlib

filename = "sample_0001.tif" #01-10 or sample.tif

img = cvlib.load(filename,0)
i = cvlib.floodFill(img, (0,0))


cvlib.displayImgs([img, i])

"""
gray = cvlib.grayscale(img)
#lap = cvlib.otsu(img, invert=True)
lap = cvlib.binaryThreshold(img, threshVal=70)
lap = cvlib.closing(lap)
lap = cvlib.opening(lap)
"""
"""
contours = cvlib.findContours(lap)
cvlib.drawContour(img, contours[0])
m = cvlib.cntInfo(img, contours[0])
poi = cvlib.poi(img, contours[0])
for key, value in poi.iteritems():
    cvlib.plotPoint(img, value, radius = 10)
print m
cvlib.displayImgs([img, lap])
"""

