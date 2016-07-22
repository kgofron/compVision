#!/usr/bin/env python
import cvlib

def work():
    img = cvlib.load("BPM1photo.jpg") # change to fetch
    contours = cvlib.findContours(img, thresh=220)
    print "Object Details:"
    m = cvlib.printCntInfo(img, contours[0])




for i in range(30):
    work()
