import cvlib
from matplotlib import pyplot as plt

angle = 0
angles = []
center = []
for i in range(24): #24
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
    #cvlib.drawContour(img, crystal, thickness=10)
    #cvlib.plotCentroid(img, crystal, radius=7)
    #cvlib.display(img)
    #cvlib.save(img, "found%d.jpg" % angle)
    angle += 15

plt.plot(angles, center, 'bo')
plt.xlabel("Angles in Degrees")
plt.ylabel("Y Pixel Coordinate of Centroid")
plt.title("Y Position of Crystal Per Angle")
plt.axis([0, 360, 0, 400])
plt.savefig("graph.png")
#plt.show()
