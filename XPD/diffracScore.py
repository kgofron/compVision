from cvlibNoEpics import *

for i in range(1):
    img = load("test%d.tif" % i, 0)
    jet = applyJET(img)
    hsv = applyHSV(img)
    #drawContours(, cnt)
    save(jet, "jet.jpg")
    save(hsv, "hsv.jpg")
