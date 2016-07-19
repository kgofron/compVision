import cvlib

img = cvlib.load("photo11.bmp")
i = cvlib.imfill(img)
i = cvlib.bitNot(i)
contours = cvlib.findContours(i)

for cnt in contours:
    cvlib.drawContour(img, cnt)
    print cvlib.cntInfo(img, cnt)
    cvlib.plotPOI(img, cnt)
cvlib.display(img)
