import cvlib

filename = "sample_0009.tif" #01-10 or sample.tif

img = cvlib.load(filename)
i = cvlib.floodFill(img, (0,0))
ope = cvlib.closing(i)
lap = cvlib.binaryThreshold(ope, threshVal=254, invert=False)
contours = cvlib.findContours(lap)
#cvlib.drawContour(img, contours[0])
m = cvlib.extremePoints(contours[0])
#m = cvlib.cntInfo(img, contours[0])
#cvlib.plotPoint(img, m["extrema"]["T"], radius=10)
print "TIP: ", m["T"]
#poi = cvlib.poi(img, contours[0])
#for key, value in poi.iteritems():
#    cvlib.plotPoint(img, value, radius = 10)
#print m
#cvlib.display(img)


