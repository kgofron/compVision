import cvlib


img = cvlib.load("coin.jpg")

co = cvlib.coherenceFilter(img)
gab = cvlib.gaborFilter(img)

cvlib.displayImgs([img, co, gab])
#cvlib.save(co, "coherence.jpg")
#cvlib.save(gab, "gaborFilter.jpg")
