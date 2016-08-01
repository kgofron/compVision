from cvlib import *

num = 24 #360/15 = 24 -> Higher number == more images, better acc
#SYS = "XF:10IDD-BI"
#DEV = "OnAxis-Cam:1"
#MOTOR = "XF:10IDD-OP{Spec:1-Ax:Th}Mtr"
#MaxAngle = 10
#MinAngle = -10
angle = 0
angleAdj = 15 #caget(MOTOR+".RBV") #XF:10IDD-OP{Spec:1-Ax:Th}Mtr.RBV
angles = []
toppt = []

# Start roatation here
for i in range(num) #and angle < MaxAngle: #change to number of images
    img = load("sample_00%d.tiff" % angle) #change name to fit img
    img = np.array(img, np.uint8)
    #angle = caget(MOTOR+".RBV") #XF:10IDD-OP{Spec:1-Ax:Th}Mtr.RVB
    angles.append(angle)
    angle += angleAdj
    flood = floodFill(img, (600,600), lo=0, hi=0)
    cnt = findContours(flood)
    top = extremePoints(cnt[0])["T"]
    print "PIN TOP %d: " % angle, top
    toppt.append(top[0])
    #plotPoint(img, top, radius = 10)
    #drawContour(img, cnt[0], thickness=10)
    #save(img, "result%d.png" % i)
    #display(img)
    

saveGraph(angles, toppt, "Pin Tip Position", "Angle", "X Coord Top") # Do We need this?
d = approxSinCurve(toppt)
print d["amplitude"], d["phase shift"], d["vertical shift"] # Extract?
saveGraph(angles, d["data"], "X Coord Per Angle", "Angles in Degrees", "X Coord Top", filename="fit.png")
makeGraph(angles, d["data"], "X Coord Per Angle", "Angles in Degrees", "X Coord Top", style="r") # Get rid of this

# use amplitude and phase to motors
