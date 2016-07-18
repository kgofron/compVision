import cvlib

slit = cvlib.load("slitHFM.jpg")
img = cvlib.load("HFM0.jpg") #change to fetch
img = cvlib.drawMatch(img, slit) # SHould probably retriueve info on pos
cvlib.display(img)
