import cvlib

angle = 0
angles = []
center = []
NumImgs = 24
angleAdj = 15

for i in range(NumImgs): #24
    img = cvlib.load("findloop_%d.jpg" % angle)
    angles.append(angle)
    rng = cvlib.inRangeThresh(img, (20,30,20), (200,130,120))
    rng = cvlib.bitNot(rng)
    cnt = cvlib.findContours(rng, thresh=250)
    if cvlib.area(cnt[0]) > cvlib.area(cnt[1]):
        crystal = cnt[0]
    else:
        crystal = cnt[1]
    centroid = cvlib.centroid(crystal)
    center.append(centroid[1])
    # NOT NEEDED
    cvlib.drawContour(img, crystal, thickness=10)
    cvlib.plotCentroid(img, crystal, radius=7)
    cvlib.display(img)
    cvlib.save(img, "found%d.jpg" % angle)
    # # # #
    angle += angleAdj

cvlib.saveGraph(angles, center, "Y Coord Per Angle", "Angle in Degrees", "Original Data Coord", [0,360,0,400], filename="graph.png")
d = cvlib.approxSinCurve(center)
print d["amplitude"], d["phase shift"], d["vertical shift"]
cvlib.saveGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid Best Fit", [0,360,0,400], style="r--", filename="fit.png")
cvlib.makeGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid", [0,360,0,400], style="r--")
# X = -[mc/pel]*A*sin(phase)
# Y = -[mc/pel]*A*cos(phase)


