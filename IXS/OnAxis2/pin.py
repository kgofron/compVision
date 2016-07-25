from cvlib import *
from matplotlib import pyplot as plt

angle = 0
angles = []
toppt = []

for i in range(24): #change to number of images
    img = load("sample_00%d.tiff" % angle) #change name to fit img
    angles.append(angle)
    flood = floodFill(img, (600,600), lo=0, hi=0)
    cnt = findContours(flood)
    #drawContour(img, cnt[0], thickness=10)
    top = extremePoints(cnt[0])["T"]
    #plotPoint(img, top, radius = 10)
    print "PIN TOP %d: " % angle, top
    toppt.append(top[0])
    #save(img, "result%d.png" % i)
    #display(img)
    angle += 15

saveGraph(angles, toppt, "Pin Tip Position", "Angle", "X Coord Top")

d = approxSinCurve(toppt)
print d["amplitude"], d["phase shift"], d["vertical shift"]
saveGraph(angles, d["data"], "Y Coord Per Angle", "Angles in Degrees", "Y Coord Centroid", filename="fit.png")
makeGraph(angles, d["data"], "Y Coord Per Angle", "Angles in Degrees", "Y Coord Centroid", style="r")
