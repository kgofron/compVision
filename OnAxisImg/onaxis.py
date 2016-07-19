import cvlib

filename = "sample_0009.tif" #01-10 or sample.tif

img = cvlib.load(filename)
k = cvlib.opening(img, kernel=(2,2))
j = cvlib.closing(k, kernel=(2,2))
cvlib.save(j, "img.jpg")
i = cvlib.floodFill(j, (700,1780), lo=20, hi=6, connectivity=4)
ope = cvlib.closing(i)
lap = cvlib.binaryThreshold(i, threshVal=254, invert=False)
contours = cvlib.findContours(lap)
cvlib.drawContour(img, contours[0])
m = cvlib.extremePoints(contours[0])
#m = cvlib.cntInfo(img, contours[0])
#cvlib.plotPoint(img, m["extrema"]["T"], radius=10)
print "TIP: ", m["T"]
#poi = cvlib.poi(img, contours[0])
#for key, value in poi.iteritems():
#    cvlib.plotPoint(img, value, radius = 10)
#print m
cvlib.displayImgs([img, j, i, ope, lap], ["img", "open/close", "flood", "clear", "bin"])


