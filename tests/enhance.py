import cvlib

img = cvlib.load("coin.jpg")

en = cvlib.enhance(img, window=10)

sh = cvlib.sharpen(img, ker=(21,21))

cvlib.displayImgs([img, en, sh])
