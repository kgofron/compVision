import cvlib

for i in range(1):
    img = cvlib.load("test%d.tif" % i)
    lap = cvlib.laplacian(img)
    #cvlib.display(lap)
    cnt = cvlib.findContours(img, thresh=250)
    #center = cnt[0]
    #for item in cnt:
    #    if cvlib.area(item) < cvlib.area(center):
    #        center = item
    cvlib.drawContours(img, cnt)
    cvlib.display(img)
