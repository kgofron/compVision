import cvlib

img = cvlib.load("comprec1-top3.jpg")
#for i in range(30):
tmp = cvlib.grayscale(img)
a = cvlib.floodFill(tmp, (2200,300), val=0, lo=1, hi=1)
a = cvlib.opening(a)
a = cvlib.bitNot(a)
m = cvlib.minMaxLoc(tmp)
print m
cvlib.plotPoints(img, [m["minLoc"], m["maxLoc"]], radius = 100)
#tmp = cvlib.binaryThreshold(tmp, threshVal=80)
cvlib.matplotlibDisplay(a)
cvlib.save(a, "gray.jpg")
