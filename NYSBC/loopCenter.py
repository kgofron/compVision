import cvlib
import numpy as np
import urllib2
import telnetlib

"""
SYS = "XF:17IDB-ES:AMX"
DEV = "Cam:7"
MOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:O}Mtr"
YMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PY}Mtr"
ZMOTOR = "XF:17IDB-ES:AMX{Gon:1-Ax:PZ}Mtr"
"""

URL = "http://192.168.1.63/axis-cgi/jpg/image.cgi"
TELNETHOST = "192.168.1.35"
YMTRSCALE = 0.859375 #100.0/120.0
ZMTRSCALE = 0.849609375 #100.0/120.0
NumImgs = 120 

tn = telnetlib.Telnet(TELNETHOST)
tn.write("RPA\r")
angle = int(tn.read_some()[:-2])  / 100.0
print angle
MaxAngle = angle + 360 #1000000
MinAngle = -180
angles = []
center = []

tn.write("SHA\r")
tn.read_some()
tn.write("PRA=36000\r")
tn.read_some()
tn.write("BGA\r")
tn.read_some()

for i in range(NumImgs):
    if angle >= MaxAngle: 
        break
    tn.write("RPA\r")
    angle = float(int(tn.read_some()[:-2])) / 100.0 
    with open('image.jpg', 'w') as f:
        f.write(urllib2.urlopen(URL).read())
    
    img = cvlib.load("image.jpg", 0) 
    #img = np.array(img, np.uint8)
    
    angles.append(angle)
    rng = cvlib.adaptiveGaussianThreshold(img)

    #pts = cvlib.templateMatchSingle(img, cvlib.load("crop.jpg"))
    #rng = cvlib.drawMatch(img, cvlib.load("crop.jpg"))
    #rng = cvlib.binaryThreshold(img, threshVal=cvlib.meanVal(img)+40)
    #rng = cvlib.inRangeThresh(img, 121, 130)
    #cvlib.display(rng)
    #edge = cvlib.floodFill(img,(0,0), lo=0, hi=1)
    #cvlib.display(edge)
    #rng = cvlib.bitNot(rng)

    cnt = cvlib.findContours(rng, thresh=250)
    cnt = cvlib.filterCnts(cnt, threshold=10)
    loop = cnt[0]
    for item in cnt:
        ext = cvlib.extremePoints(item)
        if cvlib.extremePoints(loop)["R"][0] < ext["R"][0]:
            loop = item
    crystal = loop #cnt[0]
    """if cvlib.area(cnt[0]) > cvlib.area(cnt[1]):
        crystal = cnt[0]
    else:
        crystal = cnt[1]"""
    centroid = cvlib.centroid(crystal)
    center.append(centroid[1])
    print i, angle, centroid
    cvlib.drawContour(img, crystal, thickness=2)
    cvlib.plotCentroid(img, crystal, radius=10)
    #cvlib.save(img, "found%.2f.jpg" % angle)
    #cvlib.display(img)


d = cvlib.approxSinCurve(angles, center)
print d["amplitude"], d["phase shift"], d["vertical shift"] # Phase Shift in degrees


#cvlib.caput(YMOTOR+".RLV", -YMTRSCALE*d["amplitude"]*np.sin(d["phase shift"]*np.pi/180.0))
#cvlib.caput(ZMOTOR+".RLV", -ZMTRSCALE*d["amplitude"]*np.cos(d["phase shift"]*np.pi/180.0))

tn.close()
# X = -[mc/pel]*A*sin(phase)
# Y = -[mc/pel]*A*cos(phase)


