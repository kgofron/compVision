import cvlib

img = cvlib.fetchImg("XF:10IDD-BI", "Mir:HFM-Cam:1")
img = cvlib.bitNot(img)
cvlib.display(img)

etc = cvlib.load("fetchImg.png")
etc = etc*255
cvlib.display(etc)
