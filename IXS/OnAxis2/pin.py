from cvlib import *
from matplotlib import pyplot as plt

xvals = []
yvals = []

for i in range(48, 53):
    img = load("sample_00%d.tiff" % i)
    flood = floodFill(img, (600,600), lo=0, hi=0)
    cnt = findContours(flood)
    drawContour(img, cnt[0], thickness=10)
    top = extremePoints(cnt[0])["T"]
    plotPoint(img, top, radius = 10)
    print "PIN TOP: ", top
    xvals.append(top[0])
    yvals.append(top[1])
    save(img, "result%d.png" % i)
    img = load("result%d.png" % i)
    display(img)

saveGraph(xvals, yvals, "Pin Tip Position", "X Coord", "Y Coord")
