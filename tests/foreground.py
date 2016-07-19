import cvlib

img = cvlib.load("coin.jpg")
out = cvlib.imfill(img)
cnts = cvlib.findContours(cvlib.bitNot(out))
print cvlib.cntInfo(img, cnts[0])
cvlib.drawContours(img, cnts)
cvlib.plotPOI(img, cnts[0], radius=10)
cvlib.drawContours(out, cnts)
cvlib.plotPOI(out, cnts[0], radius=10)

print "\n\n"

img2 = cvlib.load("2coins.jpg")
out2 = cvlib.imfill(img2)
cnts = cvlib.findContours(cvlib.bitNot(out2))
for cnt in cnts:
    print cvlib.cntInfo(img2, cnt)
cvlib.drawContours(img2, cnts)
cvlib.plotPOI(img2, cnts[0], radius=10)
cvlib.plotPOI(img2, cnts[1], radius=10)
cvlib.drawContours(out2, cnts)
cvlib.plotPOI(out2, cnts[0], radius=10)
cvlib.plotPOI(out2, cnts[1], radius=10)

cvlib.displayImgs([img, out, img2, out2])
