from cvlibNoEpics import *

for i in range(1):
    img = load("test%d.tif" % i, 0)
    
    jet = applyJET(img)
    hsv = applyHSV(img)
    img = drawHoughCircles(img, param1=20, param2=20, minRadius=50, maxRadius=700)
    #drawContours(, cnt)
    display(img)
