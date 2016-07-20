import cvlib

img = cvlib.fetchImg("XF:10IDD-BI", "Mir:HFM-Cam:1")
img = cvlib.bitNot(img)
cvlib.display(img)
