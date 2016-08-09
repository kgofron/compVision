import cvlib

angle = 0

for i in range(24): #24
    img = cvlib.load("findloop_%d.jpg" % angle)
    #fill = cvlib.imfill(img, threshVal= 127)
    fill = cvlib.floodFill(img, (0,0), lo=0, hi=3)
    fill = cvlib.floodFill(fill, (fill.shape[0],0), lo=3, hi=4)
    #rng = cvlib.inRangeThresh(img, (20,30,20), (200,130,120))
    #rng = cvlib.bitNot(rng)
    cvlib.display(fill)
    crystal = cvlib.findContours(fill, thresh=250)[0]
    centroid = cvlib.centroid(crystal)
    cvlib.drawContour(img, crystal, thickness=10)
    cvlib.plotCentroid(img, crystal, radius=7)
    cvlib.display(img)
    #cvlib.save(img, "loop%d.jpg" % angle)
    angle += 15
