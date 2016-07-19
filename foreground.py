import cvlib

img = cvlib.load("coin.jpg")
out = cvlib.extractForeground(img)
cvlib.display(out)
