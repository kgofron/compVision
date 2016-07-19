import cvlib

imgL = cvlib.load("mapL.jpg")
imgR = cvlib.load("mapR.jpg")
ste = cvlib.depthMap(imgL, imgR)
cvlib.display(ste)
