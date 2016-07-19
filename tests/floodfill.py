import cvlib

img = cvlib.load("selfie.jpg", flag=0)
lap = cvlib.binaryThreshold(img)
tmp = cvlib.floodFill(lap, (0,0), newval=100)
cvlib.displayImgs([lap, tmp])
