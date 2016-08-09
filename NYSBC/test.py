import urllib2
import cvlib

URL = "http://192.168.1.63/axis-cgi/jpg/image.cgi"
with open('image.jpg', 'w') as f:
    f.write(urllib2.urlopen(URL).read())

img = cvlib.load("image.jpg")
cvlib.display(img)
