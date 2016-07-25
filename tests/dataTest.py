from cvlibNoEpics import *

img8 = grayscale(load("dataRGB.jpg"))



print img8.dtype
img16 = load("bit16.png",-1)
print img16.dtype
print img16.shape

