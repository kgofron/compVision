"""
Author: William Watson
This Program splits the beam.png image
into smaller images to be worked with Scintillator.py
Image provided by my mentor, Kaz Gofron - NSLS-2 from research gate
"""

import cvlib

filename = "beam.png"
img = cvlib.load(filename)

roi = []
# first row
roi.append(cvlib.crop(img, 0, 1260,1505,2756))
roi.append(cvlib.crop(img, 1512, 1260,3019,2756))
roi.append(cvlib.crop(img, 3030, 1260,4528,2756))
roi.append(cvlib.crop(img, 4545, 1260,6044,2756))

# Second
roi.append(cvlib.crop(img, 2, 2870,1504,4348))
roi.append(cvlib.crop(img, 1520, 2870,3015,4348))
roi.append(cvlib.crop(img, 3025, 2870,4528,4348))
roi.append(cvlib.crop(img, 4545, 2870, 6044,4348))

# Third
roi.append(cvlib.crop(img, 2, 4465,1500,5964))
roi.append(cvlib.crop(img, 1512, 4465,3015,5964))
roi.append(cvlib.crop(img, 3030, 4465,4520,5964))
roi.append(cvlib.crop(img, 4550, 4465,6040,5964))

cvlib.saveImgs(roi)