from cvlibNoEpics import *

for i in range(8):
    img = load("test%d.tif" % i, 0)
    circle = drawHoughCircles(img)
    display(circle)
