import cvlib
import numpy as np

SYS = "XF:17IDB-ES:AMX"
DEV = "Cam:7"
MOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:O}Mtr"
YMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PY}Mtr"
ZMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PZ}Mtr"
YMTRSCALE = 100.0/120.0
ZMTRSCALE = 100.0/120.0
NumImgs = 33

angle = cvlib.caget(MOTOR+".RBV")
MaxAngle = 1000000
MinAngle = -180
angles = []
center = []

cvlib.caput(MOTOR+".VAL", 0)

while cvlib.caget(MOTOR+".DMOV") != 1:
    continue

cvlib.caput(MOTOR+".RLV", 360)

for i in range(NumImgs):
    if angle >= MaxAngle or cvlib.caget(MOTOR+".DMOV") == 1: #24
        break
    angle = cvlib.caget(MOTOR+".RBV")
    img = cvlib.fetchImg(SYS, DEV) 
    img = np.array(img, np.uint8)
    angles.append(angle)
    rng = cvlib.inRangeThresh(img, (0,0,0), (220,220,220))
    rng = cvlib.bitNot(rng)
    cnt = cvlib.findContours(rng, thresh=250)
    crystal = cnt[0]
    """if cvlib.area(cnt[0]) > cvlib.area(cnt[1]):
        crystal = cnt[0]
    else:
        crystal = cnt[1]"""
    centroid = cvlib.centroid(crystal)
    center.append(centroid[1])
    #cvlib.drawContour(img, crystal, thickness=10)
    #cvlib.plotCentroid(img, crystal, radius=7)
    #cvlib.save(img, "foundEPICS%d.jpg" % angle)

#cvlib.saveGraph(angles, center, "Y Coord Per Angle", "Angle in Degrees", "Original Data Coord", filename="graph.png")
d = cvlib.approxSinCurve(angles, center)
#print d["amplitude"], d["phase shift"], d["vertical shift"]

#cvlib.saveGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid Best Fit", style="r--", filename="fit.png")
#cvlib.makeGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid", style="r--")



cvlib.caput(YMOTOR+".RLV", -YMTRSCALE*d["amplitude"]*np.sin(d["phase shift"]*np.pi/180.0))
cvlib.caput(ZMOTOR+".RLV", -ZMTRSCALE*d["amplitude"]*np.cos(d["phase shift"]*np.pi/180.0))


cvlib.caput(MOTOR+".VAL", 0)
while cvlib.caget(MOTOR+".DMOV") != 1:
    continue

# X = -[mc/pel]*A*sin(phase)
# Y = -[mc/pel]*A*cos(phase)


