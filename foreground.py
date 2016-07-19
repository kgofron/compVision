import cvlib

img = cvlib.load("coin.jpg")
out = cvlib.imfill(img)


img2 = cvlib.load("2coins.jpg")
out2 = cvlib.imfill(img2)

cvlib.displayImgs([img, out, img2, out2])
