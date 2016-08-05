import cvlib
import numpy as np

SYS = "XF:17IDB-ES:AMX"
DEV = "Cam:7"
MOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:O}Mtr"
YMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PY}Mtr"
ZMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PZ}Mtr"

YMTRSCALE = 0.859375 #100.0/120.0
ZMTRSCALE = 0.849609375 #100.0/120.0
NumImgs = 120

#angle = cvlib.caget(MOTOR+".RBV")
MaxAngle = 1000000
MinAngle = -180
angles = [0,1,8,15,30,40,49,65,74,89,99,109,122,134,147,159,167,177,187,197,212,222,232,247,257,267,282,292,302,318,328,339,349,355,359]
area = []

#cvlib.caput(MOTOR+".VAL", 0)

#while cvlib.caget(MOTOR+".DMOV") != 1:
#    continue

#cvlib.caput(MOTOR+".RLV", 360)

for i in range(35): #NumImgs
    
    #if angle >= MaxAngle:# or cvlib.caget(MOTOR+".DMOV") == 1: #24
    #    break
    
    #angle = cvlib.caget(MOTOR+".RBV")
    img = cvlib.load("EPICS%03d.jpg" % angles[i]) #cvlib.fetchImg(SYS, DEV) 
    #img = np.array(img, np.uint8)
    
    #angles.append(angle)
    rng = cvlib.inRangeThresh(img, (0,0,0), (220,220,220))
    rng = cvlib.bitNot(rng)
    cnt = cvlib.findContours(rng, thresh=250)
    crystal = cnt[0]
    a = cvlib.area(crystal)
    area.append(a)
    print i, angles[i], a
    #cvlib.drawContour(img, crystal, thickness=10)
    #cvlib.plotCentroid(img, crystal, radius=7)
    #cvlib.save(img, "foundEPICS%.2f.jpg" % angle)


cvlib.saveGraph(angles, area, "Crystal Area Per Angle", "Angle in Degrees", "Area", filename="area.png")
d = cvlib.approxSinCurve(angles, area, exp=2)
print d["amplitude"], d["phase shift"], d["vertical shift"]

cvlib.saveGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid Best Fit", style="r--", filename="areaFit.png")
cvlib.makeGraph(angles, d["data"], "Y Coord Per Angle", "Angle in Degrees", "Y Coord Centroid", style="r--")



#cvlib.caput(YMOTOR+".RLV", -YMTRSCALE*d["amplitude"]*np.sin(d["phase shift"]*np.pi/180.0))
#cvlib.caput(ZMOTOR+".RLV", -ZMTRSCALE*d["amplitude"]*np.cos(d["phase shift"]*np.pi/180.0))



#cvlib.caput(MOTOR+".VAL", 0)
#while cvlib.caget(MOTOR+".DMOV") != 1:
#    continue


# X = -[mc/pel]*A*sin(phase)
# Y = -[mc/pel]*A*cos(phase)


