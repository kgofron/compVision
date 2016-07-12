import cvlib

filename = "crystals.png"
img = cvlib.load(filename)

roi = []
# first row
roi.append(cvlib.crop(img, 10, 1260,1470,2650))
roi.append(cvlib.crop(img, 1530, 1260,2900,2650))
roi.append(cvlib.crop(img, 3050, 1260,4500,2650))
roi.append(cvlib.crop(img, 4570, 1260,6020,2650))

# Second
roi.append(cvlib.crop(img, 10, 2880,1470,4300))
roi.append(cvlib.crop(img, 1530, 2880,2900,4300))
roi.append(cvlib.crop(img, 3050, 2880,4500,4300))
roi.append(cvlib.crop(img, 4570, 2880,6020,4300))

# Third
roi.append(cvlib.crop(img, 10, 4490,1470,5900))
roi.append(cvlib.crop(img, 1530, 4490,2900,5900))
roi.append(cvlib.crop(img, 3050, 4490,4500,5900))

cvlib.saveImgs(roi)