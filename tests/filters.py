import cvlib


img = cvlib.load("selfie.jpg")

co = cvlib.coherenceFilter(img)
gab = cvlib.gaborFilter(img)

cvlib.displayImgs([img, co, gab])
cvlib.save(co, "coherence.jpg")
cvlib.save(gab, "gaborFilter.jpg")
