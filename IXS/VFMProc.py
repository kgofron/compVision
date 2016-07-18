import cvlib

slit = cvlib.load("slitVFM.jpg")
img = cvlib.load("VFM0.jpg") # Change to fetch
img = cvlib.drawMatch(img, slit) # SHould probably retriueve info on pos
cvlib.display(img)
