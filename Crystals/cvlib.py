"""
BNL NSLS-2 Computer Vision Module
Author: William Watson
Date: 2016-7-14

This Module contains numerous functions designed to 
help in the data manipulation of images on the Beam-lines within NSLS-2.
It Uses the numpy and cv2 libraries to construct a simple module to 
make image processing and computer vision easier, faster, and more compact.
"""

# DEPENDENCIES
import cv2
import cv2.cv as cv # SEE IF NECESSARY
import numpy as np 
import sys
import time
from matplotlib import pyplot as plt
import matplotlib # SEE IF NECESSARY
from epics import caget, caput


# VERSION
__version__ = "0.5.4.0"
__opencv__ = cv2.__version__
__npversion__ = np.version.version
__sysver__ = sys.version
__matplotlibver__ = matplotlib.__version__


# GLOBAL VARS
colorMap_flag = {"autumn":0, "bone":1, "jet":2, "winter":3, "rainbow":4, "ocean":5, "summer":6, "spring":7, "cool":8, "hsv":9, "pink":10, "hot":11}
border_flag = {"constant":0, "reflect":2, "reflect101":4, "replicate":1, "default":4, "wrap":3}
EPICSTYPE = {1 : np.uint16 , 0 : np.uint8, 2:np.uint32, 3: np.uint64}
EPICSCOLOR = {0: "gray", 1: "bayer", 2: "RBG1"}

#######################################################################################

# EXPERIMENTAL SECTION - ALL NEW METHODS GO HERE FIRST FOR TESTING
#FIX FIX FIX

# METHODS KNOWN TO BE BROKEN / Need Further Looking!!!!!!!!!

"""
Background Subtraction Methods

Params:
        * img - Image
        * flag - OPTIONAL - algorithm select; <0 - MOG2, 0- MOG, >0-GMG; def: 0

Returns:
        * Basckground Subtracted Mask
"""
def backgroundSubtract(img, flag=0):
        fgbg = cv2.BackgroundSubtractorMOG()
        fgmask = fgbg.apply(img)
        return fgmask




# Is this really enhancing? Need to figure out...
def enhance(img, window=30):
        hp = highPassFilter(img, window=window)
        tmp = grayscale(img) + laplacian(img)
        return tmp

"""
Image Segmenting - Watershed Algorithm

Params:
	* img - image

Returns:
	* Segmented img
"""
def watershed(img):
	tmp = img.copy()
	gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
	sure_bg = cv2.dilate(opening, kernel, iterations=3)
	dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)
	ret, markers = cv2.connectedComponents(sure_fg) #IS THIS REALLY NOT IMPLEMENTED IN PYTHON?
	markers = markers+1
	markers[unknown==255] = 0
	markers = cv2.watershed(tmp, markers)
	tmp[markers == -1] = [255,0,0]
	return tmp


"""
Harris Corner Detection with SubPixel Accuracy

Params:
	* img - image
	* blockSize - OPTIONAL - size of neighborhood considered for corner detection
	* ksize - OPTIONAL - Aperture parameter of Sobel derivative used
	* k - OPTIONAL - Harris detector free parameter in equation
	* color1 - OPTIONAL - def is (0,0,255) 
	* color1 - OPTIONAL - def is (0,255,0) 

Returns:
	* Image with corners marked.
"""
def drawHarrisSubPixel(img, blockSize=2, ksize=3, k=0.04, color1=(0,0,255), color2=(0,255,0)):
	tmp = img.copy()
	gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, blockSize, ksize, k)
	dst = cv2.dilate(dst, None)
	ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
	res = np.hstack((centroids, corners))
	res = np.int0(res)
	tmp[res[:,1], res[:,0]] = color1
	tmp[res[:,3], res[:,2]] = color2
	return tmp


"""
Returns a depth map of an img

Params:
	* imgL - first image in pair
	* imgR - second image in pair
	* ndisparities - OPTIONAL - def: 16
	* blockSize - OPTIONAL - def:16

Returns:
	* Depth map of an image
"""
def depthImg(imgL, imgR, ndisparities=16, blockSize=16):
	stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=ndisparities, SADWindowSize=blockSize)
	disparity = stereo.compute(imgL, imgR)
	return disparity


"""
Returns a sharpebded image using the Unsharp Img Algorithm

Params:
        * img - image
        * ker - OPTIONAL - kernel size tuple; def:(9,9)
        * sigX - OPTIONAL - Gaussian kernel standard deviation in X dir; def:10.0

Returns:
        * sharpened img
"""
def sharpen(img, ker = (9,9), sigX=10.0):
        gaus = cv2.GaussianBlur(img, ker, sigX)
        unsharp = cv2.addWeighted(img, 1.5, gaus, -0.5, 0, img)
        return unsharp

###EXPERIMETNAL###
def fetchImgEXP(SYS, DEV):
        SYSDEV = str(SYS) + "{" + str(DEV) + "}"
        data = caget(SYSDEV + "image1:ArrayData")
        rows = caget(SYSDEV + "image1:ArraySize1_RBV")
        cols = caget(SYSDEV + "image1:ArraySize0_RBV")
        dtype = caget(SYSDEV + "cam1:DataType_RBV")
        color = caget(SYSDEV + "cam1:ColorMode_RBV")
        count = 0
        img = []
        row = []
        dtype = EPICSTYPE[caget(SYSDEV + "cam1:DataType_RBV")]
        #print dtype
        color = caget(SYSDEV + "cam1:ColorMode_RBV")
        #print color
        for i in range(rows):
                for j in range(cols):
                        row.append(data[count])
                        count = count + 1
                r = np.array(row, dtype)
                img.append(r)
                row = []
        npra = np.array(img, dtype)
        #display(npra)
        save(npra, "fetchImg.jpg")
        img = load("fetchImg.jpg") #, getColorFlag(color))
        return img


################################################################################


"""
Flood Fill Algorithm

Params:
	* img - image
	* seedPoint - startPoint
	* val -OPTIONAL - New Value; 255,255,255
        * lo - OPTIONAL - Max lower birghtness/color diff; def: 20
        * hi - OPTIONAL - Max upper birghtness/color diff; def: 20
        * fixedRng - OPTIONAL - TRUE=FIXED diff btw curr and see; FALSE=MASK only fills mask def:False
        * connectivity - OPTIONAL - 4 or 8 bit neightborrhood, def:8

Returns:
        * Flood Filles Img
"""
def floodFill(img, seedPoint, val=(255,255,255), lo=25, hi=25, fixedRng=False, connectivity=8):
        flooded = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2,w+2), np.uint8)
        flags = connectivity
        if fixedRng:
                flags |= cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(flooded, mask, seedPoint, val, (lo,)*3, (hi,)*3, flags)
        return flooded


"""
Imfill - Creates a mask for an image by removing holes, Isolates shape of object

Params:
        * img - image
        * threshVal - OPTIONAL - Threshold Value; def: 220

Returns:
        * Image with Holes Filled
"""
def imfill(img, threshVal = 220):
        tmp = grayscale(img)
        ret, thresh = cv2.threshold(tmp, threshVal, 255, cv2.THRESH_BINARY_INV)
        flood = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2,w+2), np.uint8)
        cv2.floodFill(flood, mask, (0,0), 255)
        invert = cv2.bitwise_not(flood)
        output = thresh | invert
        return output


"""
Performs a Fourier Transform using OpenCV Methods

Params: 
        * img - Image
       
Returns: 
        * Magnitude Spectrum of Image 
"""
def fourierCV(img):
	gray = grayscale(img)
	dft = cv2.dft(np.float32(gray), flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
        return magnitude_spectrum


"""
Performs a Low Pass Filter Operation on the Image

Params:
        * img - image
        * window - OPTIONAL - window size used for masking in spectrum; def: 30 - results in 60x60 window

Returns:
        * Low Pass Filtered Image
"""
def lowPassFilter(img, window=30):
        gray = grayscale(img)
	dft = cv2.dft(np.float32(gray), flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	rows, cols = gray.shape
	crow, ccol = rows/2, cols/2
	mask = np.zeros((rows, cols, 2), np.uint8)
	mask[crow-window:crow+window, ccol-window:ccol+window] = 1
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
	return img_back


"""
Performs a Fourier Transform using CV
This is a wrapper fourier function for generic fourier. 
More specific fncs are provided as well.

Params: 
        * img - Image
       
Returns: 
        * Magnitude Spectrum of Image 
"""
def fourier(img):
        return fourierCV(img)


"""
Performs a Fourier Transform using Numpy Methods

Params: 
        * img - Image
       
Returns: 
        * Magnitude Spectrum of Image 
"""
def fourierNP(img):
	gray = grayscale(img)
	f = np.fft.fft2(gray)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift)) # RETURN THIS
        return magnitude_spectrum


"""
Performs a High Pass Filter Operation on the Image

Params:
        * img - image
        * window - OPTIONAL - window size used for masking in spectrum; def: 30 - results in 60x60 window

Returns:
        * High Pass Filtered Image
"""
def highPassFilter(img, window=30):
        gray = grayscale(img)
	f = np.fft.fft2(gray)
	fshift = np.fft.fftshift(f)
	rows, cols = gray.shape
	crow, ccol = rows/2, cols/2
	fshift[crow-window:crow+window, ccol-window:ccol+window] = 0
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)
	return img_back


"""
Displays an image using MatPlotLib
Useful for displaying all images and Magnitude Spectrums

Params:
        * img - image
        * title - OPTIONAL - name of Image
        * colorFlag - OPTIONAL - color flag for imshow; def: 'gray'
"""
def matplotlibDisplay(img, title="Image", colorFlag = 'gray'):
        plt.imshow(img, colorFlag)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.show()


"""
Displays multiple images in a matplotlib window

Params: 
        * img - image
        * title - OPTIONAL - name of Image
        * colorFlag - OPTIONAL - color flag for imshow; def: 'gray'
"""
def matplotlibDisplayMulti(imgs, titles=None, colorFlag='gray'):
        if titles is None:
                titles = []
                for i in range(len(imgs)):
                        titles.append("IMAGE " + str(i))
        for i in range(len(imgs)):
                plt.subplot(1, len(imgs),  1+i)
                plt.imshow(imgs[i], colorFlag)
                plt.title(titles[i])
                plt.xticks([])
                plt.yticks([])
        plt.show()


"""
Prints the version codes for cvlib, OpenCV, and Numpy. For Refrence
"""
def version():
        print "Cvlib  Version: " + str(__version__)
        print "OpenCV Version: " + str(__opencv__)
        print "Numpy  Version: " + str(__npversion__)
        print "Matplotlib Ver: " + str(__matplotlibver__)
        print "Python Version: " + str(__sysver__)

        
"""
Transposes a matrix

Params:
	* matrix - matrix

Returns:
	* transpose of matrix
"""
def transpose(matrix):
	return cv2.transpose(matrix)


"""
Returns a dictionary into two lists of [[key][values]]
"""
def dictToLst(dictionary):
        keys = []
        values = []
        for key, value in dictionary.iteritems():
                keys.append(key)
                values.append(value)
        return [keys, values]


"""
Turns two lists into a dictionary

Params:
        * key - List of Keys
        * value - List of Values

Returns:
        * Dictionary of Key/Values
"""
def lstToDict(key, value):
        return dict(zip(key, value))


"""
Returns the Mean Color (Regular Img) / Mean Intensity (Grayscale)

Params:
	* img - image

Returns:
	* Mean Color / Mean Image
"""
def meanVal(img):
	mean = cv2.mean(img)
	if img is None:
		print "ERROR: MeanValue: Sent in None-Type Object"
		return -1
	if len(img.shape) == 3:
		return (mean[0], mean[1], mean[2])
	elif len(img.shape) == 2:
		return (mean[0])
	else:
		return mean


"""
Flips a 2D array around vertical, horizontal, or both axes

Params:
	* img - image
	* code - OPTIONAL - flip code
						flip code == 0 -> Vertical
						flip code  > 0 -> Horizontal
						flip code  < 0 -> Diagonal

Returns:
	* flipped image
"""
def flip(img, code=0):
	return cv2.flip(img, flipCode=code)


"""
Finds the mask for an image

Params:
	* img - image
	* invert - OPTIONAL - def False; if def, returns mask, if True, returns bitwise not of mask

Returns:
	* mask
"""
def mask(img, invert=False):
	imgray = grayscale(img) #.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	if not invert:
		return mask
	else:
		return mask_inv


"""
Copy an image, removes all refrences to original image

Params: 
	* img- image to be copied

Returns:
	* copied img
"""
def copy(img):
	return img.copy()


"""
Removes contours that contain less points than threshold, EXCLUSIVE

Params:
	* cnts - list of contours
	* threshold - OPTIONAL - Removes all contours less than it, 
							 i.e only allows for contours greater than it; 
							 def: 5

Returns:
	* new list of contours filtered
"""
def filterCnts(cnts, threshold = 5):
	c = []
	for item in cnts:
		if threshold < len(item):
			c.append(item)
	return c


"""
Returns pertinent information regarding a contour

Params:
	* img - image
	* cnt - contour

Returns:
	* dictionary of cnt info
"""
def cntInfo(img, cnt):
	pts = extremePoints(cnt)
	roi = crop(img, pts["L"][0], pts["T"][1], pts["R"][0], pts["B"][1])
	m = minMaxLoc(roi)
	m["minLoc"] = (m["minLoc"][0] + pts["L"][0], m["minLoc"][1] + pts["T"][1])
	m["maxLoc"] = (m["maxLoc"][0] + pts["L"][0], m["maxLoc"][1] + pts["T"][1])
	cross = abs(pts["L"][0] - pts["R"][0])
	height = abs(pts["T"][1] - pts["B"][1])
	cent = centroid(cnt)
	angle = orientation(cnt)
	areaVal = area(cnt)
	per = perimeter(cnt)
	ar = aspectRatio(cnt)
	ext = extent(cnt)
	sold = solidity(cnt)
	eqD = equivalentDiameter(cnt)
	me = meanVal(grayscale(roi))
	su = sumPixel(grayscale(roi))
	d = {"sum intensity":su, "mean intensity":me, "area":areaVal, "perimeter":per, "aspect ratio":ar, "extent":ext,"solidity":sold, "equivalent diameter":eqD, "width": cross, "height" : height, "centroid" : cent, "extrema" : pts, "min":m["minLoc"], "max":m["maxLoc"], "orientation" : angle}
	return d


"""
Sums all of the pixels in an image
Best for Grayscale

Params:
	* image

Returns:
	* sum of pixel values
"""
def sumPixel(img):
	if img is None:
		print "ERROR: SumPixel: None-Type Object sent in"
		return -1
	sh = img.shape
	su = 0
	for i in range(0, sh[0]):
		for j in range(0, sh[1]):
			su = su + img[i][j]
	return su


"""
Returns only points of interest for a contour

Params:
	* img - image
	* cnt- contour

Returns:
	* points of interest (max/min, extremas, center)
"""
def poi(img, cnt):
	m = cntInfo(img, cnt)
	d = {"max":m["max"],"B":m["extrema"]["B"],"T":m["extrema"]["T"],"R":m["extrema"]["R"],"L":m["extrema"]["L"],"min":m["min"],"centroid":m["centroid"]}
	return d


"""
Returns only scalar information for a contour

Params:
	* img- image
	* cnt- contour

Returns:
	* scalar contour information
"""
def scalarInfo(img, cnt):
	m = cntInfo(img, cnt)
	d = {"perimeter":m["perimeter"], "oreientation":m["orientation"], "solidity":m["solidity"],"height":m["height"], "extent":m["extent"], "aspect ratio":m["aspect ratio"], "area":m["area"], "sum intensity":m["sum intensity"], "width":m["width"], "equivalent diameter": m["equivalent diameter"], "mean intensity": m["mean intensity"]}
	return d


"""
Prints the contour information line by line via key: value format

Params:
        * image - image
        * cnt - contour
""" 
def printCntInfo(img, cnt):
        m = cntInfo(img, cnt)
        lst = dictToLst(m)
        for i in range(len(lst[0])):
                print str(lst[0][i]) + ": " + str(lst[1][i])
                

"""
Prints a Dictionary in Key: Value format

Params:
        * m - Dictionary to print
"""
def printDic(m):
        lst = dictToList(m)
        for i in range(len(lst[0])):
                print str(lst[0][i]) + ": " + str(lst[1][i])


"""
Plots the points of interest of a cnt

Params:
	* img - image
	* cnt - contour
	* radius - OPTIONAL - radius of point; def: 3
	* color - OPTIONAL - color of pt; def: (100, 100, 255)
"""
def plotPOI(img, cnt, radius = 3, color=(100,100,255)):
	m = poi(img, cnt)
	for key, value in m.iteritems():
		plotPoint(img, value, radius = radius, color = color)


"""
Returns a list of all cnt info for a list of contours
Here, cnts[0] -> cntProp[0]; i.e. all contour indexes map to info index

Params:
	* img - image containing contours
	* cnts - list of contours

Returns:
	* contour information in list format
"""
def cntInfoMult(img, cnts):
	cntProp = []
	for item in cnts:
		cntProp.append(cntInfo(img, item))
	return cntProp


"""
Finds all the points that comprises an object (Pixel Points)

Params:
	* img - image
	* cnt - contour

Returns:
	* pixelpoints
"""
def pixelPoints(img, cnt):
	m = np.zeros(grayscale(img).shape, np.uint8)
	cv2.drawContours(m, [cnt], 0, 255, -1)
	pixelpoints = cv2.findNonZero(m)
	return pixelpoints


"""
Fill a Contour - Fills a contour with color onto img

Params:
	* img - image
	* cnt - contour
	* color - OPTIONAL - color of fill, def: (255,255,0)
"""
def fillContour(img, cnt, color = (255,255,0)):
	cv2.drawContours(img, [cnt], 0, color, -1)


"""
Returns the Max Val, Min Val, and Locations

Params:
	* img - image

Returns:
	* (minVal, maxVal, minLoc, maxLoc) in dicitonary format
"""
def minMaxLoc(img):
	maskVar = mask(img)
	pt = cv2.minMaxLoc(grayscale(img), maskVar)
	d = {"minVal": pt[0], "maxVal":pt[1], "minLoc":pt[2], "maxLoc":pt[3]}
	return d


"""
Returns the Max value and Location

Params:
	* img - image

Returns:
	* (MaxVal, MaxLoc) in dictionary format
"""
def findMax(img):
	d = minMaxLoc(img)
	return {"maxVal":d["maxVal"], "maxLoc":d["maxLoc"]}


"""
Returns the Min value and Location

Params:
	* img - image

Returns:
	* (MinVal, MinLoc) in dictionary format
"""
def findMin(img):
	d = minMaxLoc(img)
	return {"minVal":d["minVal"], "minLoc":d["minLoc"]}


"""
Image Addition - add two images
	
Params:
	* img1 - image 1
	* img2 - image 2 or a scalar quantity

Returns:
	* Addition of img1 and img2
"""
def add(img1, img2):
	return cv2.add(img1, img2)


"""
Image Blending - Added Weighting

Params:
	* img1 - first image
	* wt1 - weight of first image
	* img2 - second img
	* wt2 - weight of second image
	* gamma - OPTIONAL - def is 0

Returns:
	* weighted addition of imgs
"""
def addWeight(img1, wt1, img2, wt2, gamma=0):
	dst = cv2.addWeight(img1, wt1, img2, wt2, gamma)
	return dst


"""
Bitwise Ops - And

Params:
	* img1 - first input array/scalar
	* img2 - OPTIONAL - second input array/scalar; def is usually img1
	* maskVar - OPTIONAL - mask of img; def of None

Returns:
	* Bitwise And of imgs
"""
def bitAnd(img1, img2=None, maskVar = None):
	if img2 is None:
		img2 = img1
	return cv2.bitwise_and(img1, img2, mask=maskVar)


"""
Bitwise Ops - Or

Params:
	* img1 - first input array/scalar
	* img2 - OPTIONAL - second input array/scalar; def is usually img1
	* mask - OPTIONAL - mask of img; def is None

Returns:
	* Bitwise Or of imgs
"""
def bitOr(img1, img2=None, mask = None):
	if img2 is None:
		img2 = img1
	return cv2.bitwise_or(img1, img2, mask=mask)


"""
Bitwise Ops - Exclusive Or

Params:
	* img1 - first input array/scalar
	* img2 - OPTIONAL - second input array/scalar; def is usually img1
	* mask - OPTIONAL - mask of img; def is None

Returns:
	* Bitwise Exclusive Or of imgs
"""
def bitXor(img1, img2=None, mask = None):
	if img2 is None:
		img2 = img1
	return cv2.bitwise_xor(img1, img2, mask=mask)


"""
Bitwise Ops - Not

Params:
	* img - input img

Returns:
	* Bitwise Not of imgs
"""
def bitNot(img):
	return cv2.bitwise_not(img)


"""
Image Dimensions

Params:
	* img - Image

Returns:
	* (x, y, c) - x pixels, y pixels, channels
"""
def shape(img):
	if len(img.shape) == 3:
		y, x, c = img.shape
		return (x, y, c)
	else:
		y, x = img.shape
		return (x, y)


"""
Image Crop: Returns a Crop Img
Can be Used for ROI

Params:
	* img - image
	* x0 - X Position (start)
	* y0 - Y Position (start)
	* x1 - X End Position
	* y1 - Y End Position

Returns:
	* imgROI - from [y:y+h, x:x+w]
"""
def crop(img, x0, y0, x1, y1):
	crop = img[y0:y1, x0:x1]
	return crop


"""
Image Cropping/ROI for PTS or Tuples as coordinates

Params:
	* img - image
	* start - (x, y) tuple for start pt
	* end - (x, y) tuple for end pt

Returns:
	* Cropped Image ROI in rectangle bounded by start:end
"""
def cropPt(img, start, end):
	x0, y0 = start
	x1, y1 = end
	return crop(img, x0, y0, x1, y1)


"""
Image ROI: Returns a Region of Interest
Can be Used for Cropping

Params:
	* img - image
	* x - X Position (start)
	* y - Y Position(start)
	* w - Width - How much region spans after X
	* h - Height - How much region spans after Y

Returns:
	* imgROI - from [x:x+w, y:y+h] 
"""
def roi(img, x, y, w, h):
	roi = img[y:y+h, x:x+w]
	return roi


"""
Crops a contour based on extrema

Params:
	* img - Image
	* cnt - contour

Returns:
	* cropped image to contour
"""
def cropCnt(img, cnt):
	pts = extremePoints(cnt)
	roi = crop(img, pts["L"][0], pts["T"][1], pts["R"][0], pts["B"][0])
	return roi


"""
Returns a hsv version of img

Params:
	* img - image

Returns:
	* HSV image if orginial is RBG
"""
def hsv(img):
	if img is None:
		print "Img is None"
		sys.exit()
	if len(img.shape) > 2:
		return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	return None


"""
Returns a grayscale version of img

Params:
	* img - image

Returns:
	* Grayscale image
"""
def grayscale(img):
	if img is None:
		print "Img is None"
		sys.exit()
	if len(img.shape) > 2:
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		return img


"""
Wrapper Function for cv2.cvtColor()
Used to change Color-space of Image
WARNING: No error checking is performed

Params:
	* img - image
	* flag - color flag, use colorFlags() to see list of flags available, cv2.FLAG

Returns:
	* Image in new colorspace
"""
def cvtColor(img, flag):
	return cv2.cvtColor(img, flag)


"""
Apply a Color map to an img

Params:
	* img - image
	* flag - string flag or corresponding int found in getColorMapFlags()

Returns:
	* img in color map
"""
def applyColorMap(img, flag):
	if isinstance(flag, basestring):
		flag = flag.lower()
		gray = grayscale(img)
		res = cv2.applyColorMap(gray, colorMap_flag[flag])
		return res
	else:
		gray = grayscale(img)
		return cv2.applyColorMap(gray, flag)


"""
Apply a JET Color Map to img

Params:
	* img - image

Returns:
	* JET img
"""
def applyJET(img):
	return applyColorMap(img, "jet")


"""
Apply a HSV Color Map to img

Params:
	* img - image

Returns:
	* HSV img
"""
def applyHSV(img):
	return applyColorMap(img, "hsv")


"""
Returns the flags and corresponding int values

Returns:
	* dicitonary of values to int codes
"""
def getColorMapFlags():
	return colorMap_flag


"""
Finds the HSV value from a BGR value, good for using ot track

Params:	
	* BGR - BGR value i.e. [X, Y, X]

Returns:
	* HSV - HSV value corresponding to it.
"""
def bgr2hsv(BGR):
	color = np.uint8([[BGR]])
	hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
	return hsv_color


"""
Returns the list of acceptable flags for color conversion 
used for change colormap fnc

Params:
	* filter - OPTIONAL - filters flags to specified string

Returns:
	* List of Color Flags
"""
def colorFlags(filterStr=""):
	filterStr = filterStr.upper()
	flags = [i for i in dir(cv2) if i.startswith('COLOR_') and filterStr in i]
	return flags


"""
Isolates a Color within an Image for tracking
Useful for tracking certain colors
Use within while loop for video

Params:
	* img - Color Image
	* lower - Lower Threshold of Color, [X, Y, Z] format
	* upper - Upper Threshold of Color, [X, Y, Z] format

Returns:
	* Isolated img
"""
def trackObject(img, lower, upper):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_col = np.array(lower)
	upper_col = np.array(upper)
	mask = cv2.inRange(hsv, lower_col, upper_col)
	res = cv2.bitwise_and(img, img, mask=mask)
	return res


"""
Thresholds an image between two color ranges, or scalars for grayscale

Params:
	* img - Image to threshold
	* lower - lower bound, either [X, Y, Z] for color or scalar for gray
	* upper - upper bound, either [X, Y, Z] for color or scalar for gray 
"""
def inRangeThresh(img, lower, upper):
	if len(img.shape) == 2 and isinstance(lower, (int, long)) and isinstance(upper, (int, long)):
		mask = cv2.inRange(img, np.array(lower), np.array(upper))
		res = cv2.bitwise_and(img, img, mask=mask)
		return res
	elif len(img.shape) == 3 and isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple)):
		lower_col = np.array(lower)
		upper_col = np.array(upper)
		mask = cv2.inRange(img, lower_col, upper_col)
		res = cv2.bitwise_and(img, img, mask=mask)
		return res
	else:
		print "ERROR: InRangeThresh: Invalid format for lower/upper"
		sys.exit()


"""
Displays an Image onto the screen and waits for user to close

Params:
	* img - image to display
        * name - OPTIONAL - string name of window, def is IMAGE
        * wait - OPTIONAL - time in ms for screen to wait, def:0 - INDEFINITE
"""
def display(img, name="IMAGE", wait=0):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.imshow(name, img)
	cv2.waitKey(wait) & 0xFF
	cv2.destroyAllWindows()


"""
Returns the centroid to a given contour

Params:
	* cnt - contour

Returns:
	* (cx, cy) -> Pixel Position of the Centroid
"""
def centroid(cnt):
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return (cx, cy)


"""
Plots a centroid onto an image from a given contour

Params:
	* cnt - contour
	* img - image to modify
	* color - OPTIONAL - specify color of centroid cross
"""
def plotCentroid(img, cnt, radius = 3, color=(255, 255, 0)):
	cx, cy = centroid(cnt)
	drawCircle(img, (cx, cy), radius = radius, color = color)
	return (cx, cy)


"""
Sorts a list of contours by number of points, in descending order

Params:
	* contours - list of contours to sort

Returns:
	* Sorted list of contour objects such that the largest contour is first
"""
def sort(contours):
	return sorted(contours, reverse=True, key=len)


"""
Returns the contours to a given image, sorted by size of contour

Params:
	* img - Image to find contours in
	* thresh - OPTIONAL - threhsold value used
	* val - OPTIONAL - value to set items in threshold

Returns:
	* Sorted List of Contours found in img
"""
def findContours(img, thresh=127, val=255):
	gray = grayscale(img)
	ret, binary = cv2.threshold(gray, thresh, val, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	errorCheckContour(contours)
	return sort(contours) 			# Should this also Return hierarchy?


"""
Error checks the contours in images and stops program if something is wrong

Params:
	* contours - list of contours to error check
"""
def errorCheckContour(contours):
	if contours is None:
		print "ERROR: FindContours: Missing objects: No contours found, please check image...\n"
		sys.exit() # maybe return false?


"""
Approimates a contour to a given epsilon

Params:
	* cnt - contour to approximate
	* epsilon - OPTIONAL- percentage of arc length allowed as max distance from contour to approx

Returns:
	* approximated contour
"""
def contourApprox(cnt, epsilon = 0.005):
	epsilon = epsilon*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)
	return approx


"""
Returns the Aspect Ratio (Ratio of width to height of bounding rectangle)

Params:
	* cnt - contour

Returns:
	* Aspect Ratio of cnt
"""
def aspectRatio(cnt):
	x, y, w, h = cv2.boundingRect(cnt)
	return float(w) / h


"""
Returns the Extent (Ratio of contour area to bounding rectangle area)

Params:
	* cnt - contour

Returns:
	* Extent of cnt
"""
def extent(cnt):
	area = cv2.contourArea(cnt)
	x, y, w, h = cv2.boundingRect(cnt)
	rect_area = w*h
	return float(area)/rect_area


"""
Calculates the up-right bounding rectangle of a point set / cnt

Params:
	* cnt - Contour

Returns:
	* Bounding Rectangle
"""
def boundingRect(cnt):
	x, y, w, h = cv2.boundingRect(cnt)
	return {"x":x, "y": y, "w": w, "h": h}


"""
Returns the points to a rectangle bounding cnt

Params:
	* cnt - Contour

Returns:
	* points to bouding rectangle
"""
def boundingRectPoints(cnt):
	x, y, w, h = cv2.boundingRect(cnt)
	first = (x, y)
	end = (x+w, y+h)
	return {"top-left": first, "bottom-right":end}


"""
Minimum Enclosing Circle

Params:
	* cnt - Contour

Returns:
	* Min Enclose Circle info
"""
def minEncloseCircle(cnt):
	(x, y), radius = cv2.minEnclosingCircle(cnt)
	center = (int(x), int(y))
	radius = int(radius)
	return {"center" : center, "radius": radius}


"""
Returns the Solidity (Ratio of area to convex hull area)

Params:
	* cnt - contour

Returns:
	* Solidity of cnt
"""
def solidity(cnt):
	area = cv2.contourArea(cnt)
	hull = cv2.convexHull(cnt)
	hull_area = cv2.contourArea(hull)
	return float(area) / hull_area


"""
Returns area of cnt

Params:
	* cnt - contour

Returns:
	* Area of cnt
"""
def area(cnt):
	return cv2.contourArea(cnt)


"""
Returns the moments of a cnt

Params:
	* cnt - contour

Returns:
	* Dictionary of moments for a cnt
"""
def moments(cnt):
	return cv2.moments(cnt)


"""
Returns the Hu Moments from moments

Params:
	* cnt - Contour

Returns:
	* Hu Moments
"""
def huMoments(cnt):
	return cv2.HuMoments(moments(cnt))


"""
Returns the Contour Perimeter

Params:
	* cnt - contour

Returns:
	* Perimeter of cnt
"""
def perimeter(cnt):
	return cv2.arcLength(cnt, True)


"""
Returns the Equivalent Diameter (Circle whose area is same as cnt area)

Params:
	* cnt = contour

Returns:
	* Equivalent Diameter
"""
def equivalentDiameter(cnt):
	return np.sqrt(4 * (cv2.contourArea(cnt)) / np.pi)


"""
Finds the nearest fitting ellippse to a contour

Params:
	* cnt - contour

Returns:
	* (center, Major, Minor, angle) - in dicitionary format
"""
def fitEllipse(cnt):
	(x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
	return {"center":(x,y), "major":MA, "minor":ma, "angle":angle}


"""
Returns the Orientation of the Object

Params:
	* cnt - Contour

Returns:
	* Angle at which object is directed
"""
def orientation(cnt):
	(x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
	return angle


"""
Returns the Major and Minor axis lengths of the nearest fitting ellipse for the contour

Params:
	* cnt - contour

Returns:
	* (Major Axis, Minor Axis) - lengths of Major and Minor Axis ellipse for a given contour
"""
def axis(cnt):
	(x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
	return {"major":MA, "minor":ma}


"""
Draws a contour onto an image

Params:
	* img - image to draw on
	* cnt - contour to draw
	* color - OPTIONAL - color of cnt, default is (0, 255, 0)
	* thickness - OPTIONAL - thickness of cnt line, default is 2
"""
def drawContour(img, cnt, color=(0, 255, 0), thickness=2):
	cv2.drawContours(img, [cnt], 0, color, thickness)


"""
Draws all contours from a list

Params:
	* img - image to draw on
	* cnt - contours to draw
	* color - OPTIONAL - color of cnt, default is (0, 255, 0)
	* thickness - OPTIONAL - thickness of cnt line, default is 2
"""
def drawContours(img, cnt, color=(0, 255, 0), thickness=2):
	cv2.drawContours(img, cnt, -1, color, thickness)


"""
Returns the Extreme Points of a contour

Params:
	* cnt - contour

Returns:
	* (LeftMost, RightMost, TopMost, BottomMost)
"""
def extremePointsTup(cnt):
	leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
	rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
	topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
	bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
	return (leftmost, rightmost, topmost, bottommost)


"""
Extreme Points Dictionary

Params:
	* cnt - contour

Returns:
	* Dictionary of extrema, such that L = Left, R = Right, T = Top, B = Bottom
"""
def extremePoints(cnt):
	pts = extremePointsTup(cnt)
	d = {"L":pts[0], "R":pts[1], "T":pts[2], "B":pts[3]}
	return d
	

"""
Crops A Contour to a ROI by using extreme points to map values

Params:
	* img - image
	* cnt - contour

Returns:
	* Crop ROI with Extrema as base points
"""
def cropCnt(img, cnt):
	d = extremePoints(cnt)
	roi = cropPt(img, (d["L"][0], d["T"][1]), (d["R"][0], d["B"][1]))
	return roi


"""
Plots a single point onto an image

Params:
	* img - image
	* point - pixel tuple to plot at (x, y)
	* radius - OPTIONAL - radius of point to be plotted
	* color - OPTIONAL - color of pixel to be plotted
"""
def plotPoint(img, point, radius = 3, color = (0, 0, 255)):
	drawCircle(img, point, radius = radius, color=color)
	

"""
Plots a list of points onto an img

Params: 
	* img - image
	* points - list of points to plot
	* radius - OPTIONAL - radius of points to be plotted
	* color - OPTIONAL - color of points to plot, default is (0, 0, 255)
"""
def plotPoints(img, points, radius = 3, color= (0, 0, 255)):
	for pt in points:
		drawCircle(img, pt, radius = radius, color = color)


"""
Draws a circle at a point with given radius onto img.

Params:
	* img - image
	* center - (x,y) center of circle
	* radius - OPTIONAL - radius of circle, default is 3
	* color - OPTIONAL - default is (0, 0, 255) - RED
	* fill - OPTIONAL - default is -1, change for outline thickness
"""
def drawCircle(img, center, radius = 3, color = (0,0,255), fill = -1):
	cv2.circle(img, center, radius, color, fill)


"""
Draws a line btw 2 points onto img.

Params:
	* img - image
	* start - start point
	* end - end point
	* color - OPTIONAL - default is (0, 0, 255) - RED
	* thickness - OPTIONAL - default is 3, change for thickness
"""
def drawLine(img, start, end, color = (0,0,255), thickness = 3):
	cv2.line(img, start, end, color, thickness)


"""
Draws a rectangle btw 2 points (Top Left and Bottom Right) onto img.

Params:
	* img - image
	* top_left - top left point of rectangle
	* bottom_right - bottom right point of rectangle
	* color - OPTIONAL - default is (0, 0, 255) - RED
	* thickness - OPTIONAL - default is 3, change for thickness
"""
def drawRectangle(img, top_left, bottom_right, color = (0,0,255), thickness = 3):
	cv2.rectangle(img, top_left, bottom_right, color, thickness)


"""
Draws an ellipse onto img.

Params:
	* img - image
	* center - center of ellipse (x,y)
	* (Major Axis Len, Minor Axis Len)
	* Angle - angle of rotation
	* startAngle - OPTIONAL - def is 0
	* endAngle - OPTIONAL - def is 360
	* color - OPTIONAL - default is (0, 0, 255) - RED
	* fill - OPTIONAL - default is -1, change for thickness
"""
def drawEllipse(img, center, axes, angle, startAngle=0, endAngle=360, color = (0,0,255), fill = -1):
	cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, fill)


"""
Draws a nearest fitting ellipse to a contour

Params:
	* img - image to draw on
	* cnt - contour
"""
def drawFitEllipse(img, cnt):
	d = fitEllipse(cnt)
	cv2.ellipse(img, d["center"], (d["major"], d["minor"]), d["angle"])


"""
Returns properties of an image in a dictionary format

Params: 
	* img - image

Returns:
	* dictionary of properties "shape, rows, columns, channels, size, dtype"
"""
def imgProp(img):
	d = {}
	d["shape"] = img.shape
	d["rows"] = img.shape[0]
	d["columns"] = img.shape[1]
	if len(img.shape) is 3:
		d["channels"] = img.shape[2]
	d["size"] = img.size
	d["dtype"] = img.dtype
	return d


"""
Returns the Number of pixels in an image

Params:
	* img - image

Returns:
	* size of image in pixels
"""
def size(img):
	return img.size


"""
Returns the moments of a contour

Params: 
	* cnt - contour

Returns:
	* dictionary of moments
"""
def moments(cnt):
	return cv2.moments(cnt)


"""
Point Polygon Test - Finds the Shortest Distance between a point in the image and a contour

Params:
	* cnt - contour
	* point - point coordinates
	* distance - OPTIONAL - True if signed distance, False for Inside/Outside/On cnt

Returns:
	* Relation of point to cnt
"""
def pointPolygonTest(cnt, point, distance = True):
	return cv2.pointPolygonTest(cnt, point, distance)


"""
Takes two images and compares them.

Params:
	* img1 - first image
	* img2 - second image

Returns:
	* metric showing similarity, lower the result, better the match
"""
def matchShapesImages(img1, img2):
	cnt1 = findContours(img1)[0]
	cnt2 = findContours(img2)[0]
	ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
	return ret


"""
Takes two contours and compares them.

Params:
	* cnt1 - first contour
	* cnt2 - second contour

Returns:
	* metric showing similarity, lower the result, better the match
"""
def matchShapes(cnt1, cnt2):
	ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
	return ret


"""
Returns a binary threshold of the image

Params:
	* img - image to threshold
	* threshVal - OPTIONAL - threshold value to classify pixels Default 127
	* maxVal - OPTIONAL - value to be given if pixel over threshVal. Default 255
	* invert - OPTIONAL - True means inverts binary threshold (white to black, vice versa), false otherwise. Default True

Returns:
	* threshold image
"""
def binaryThreshold(img, threshVal = 127, maxVal = 255, invert=True):
	gray = grayscale(img)
	if invert:
		ret, thresh = cv2.threshold(img, threshVal, maxVal, cv2.THRESH_BINARY_INV)
	else:
		ret, thresh = cv2.threshold(img, threshVal, maxVal, cv2.THRESH_BINARY)
	return thresh


"""
Returns an img after Adaptive Mean Thresholding

Params:
	* img - image

Returns: 
	* Image that has undergone Adaptive Mean Thresholding
"""
def adaptiveMeanThreshold(img):
	gray = grayscale(img)
	gray = cv2.medianBlur(gray, 5)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	return thresh


"""
Returns an img after Adaptive Gaussian Thresholding

Params:
	* img - image

Returns: 
	* Image that has undergone Adaptive Gaussian Thresholding
"""
def adaptiveGaussianThreshold(img):
	gray = grayscale(img)
	gray = cv2.medianBlur(gray, 5)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	return thresh


"""
Returns an img after Otsu Binarization

Params:
	* img - image
	* invert - OPTIONAL - uses inverse binary if True, def: False

Returns:
	* An Otsu Binarization of the img
"""
def otsu(img, invert=False):
	gray = grayscale(img)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	if invert:
		ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		return thresh
	else:
		ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		return thresh


"""
Rotates the image by a given degree

Params:
	* img - image
	* deg - degrees to rotate
	* center - OPTIONAL - coordinate to rotate around, def is center of img

Returns:
	* rotated img
"""
def rotate(img, deg, center=None):
	gray = grayscale(img)
	rows, cols = gray.shape
	if center is None:
		M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
	else:
		M = cv2.getRotationMatrix2D(center, deg, 1)
	dst = cv2.warpAffine(gray, M, (cols, rows))
	return dst 


"""
Canny Edge Detection - Returns an img with only edges

Params:
	* img - image
	* minVal - OPTIONAL - Minimum Threshold
	* maxVal - OPTIONAL - Maximum Threshold

Returns:
	* Image with only Edges
"""
def edgeDetect(img, minVal=100, maxVal=200):
	gray = grayscale(img)
	edges = cv2.Canny(gray, minVal, maxVal, True)
	return edges


"""
Histogram Equalization - Improves the contrast of an image via global contrast

Params:
	* img - image

Returns:
	* Equalized Image
"""
def eqHist(img):
	gray = grayscale(img)
	equ = cv2.equalizeHist(gray)
	return equ


"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) - Improves contrast via Adaptive Histogram Equalization

Params:
	* img - image
	* clipLimit - OPTIONAL - contrast limit to clip pixels before hist eq, def is 2.0
	* tileGridSize - (row, col) blocks that are equalized - def is (8,8)

Returns:
	* Equalized Image
"""
def adaptiveEqHist(img, clipLimit=2.0, tileGridSize=(8,8)):
	gray = grayscale(img)
	clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
	cl1 = clahe.apply(gray)
	return cl1


"""
Loads an image from a file name and returns it in color mode (default).
Safety check if img fails to load.

Params:
	* filename - filename of image
	* flag - OPTIONAL - open mode: 1 = COLOR, 0 = GRAYSCALE, -1 = UNCHANGED

"""
def load(filename, flag=None):
	if flag is None:
		img = cv2.imread(filename)
	elif flag is 1 or flag is 0 or flag is -1:
		img = cv2.imread(filename, flag)
	else:
		print "ERROR: Load: Incorrect flag parameter: " + str(flag) + "\n"
		sys.exit()
	if img is None:
		print "ERROR: Load: Image not found/supported at: " + str(filename) + "\n"
		sys.exit()
	else:
		return img

        
"""
Loads an image from an EPICS PV Value

Params:
    * SYS - System String
    * DEV - Device String

Returns:
    * Loaded Image
"""
def fetchImg(SYS, DEV):
        SYSDEV = str(SYS) + "{" + str(DEV) + "}"
        data = caget(SYSDEV + "image1:ArrayData")
        rows = caget(SYSDEV + "image1:ArraySize1_RBV")
        cols = caget(SYSDEV + "image1:ArraySize0_RBV")
        dtype = caget(SYSDEV + "cam1:DataType_RBV")
        color = caget(SYSDEV + "cam1:ColorMode_RBV")
        count = 0
        img = []
        row = []
        dtype = EPICSTYPE[caget(SYSDEV + "cam1:DataType_RBV")]
        color = caget(SYSDEV + "cam1:ColorMode_RBV")
        for i in range(rows):
                for j in range(cols):
                        row.append(data[count])
                        count = count + 1
                r = np.array(row, dtype)
                img.append(r)
                row = []
        npra = np.array(img, dtype)
        save(npra, "fetchImg.jpg")
        img = load("fetchImg.jpg")
        return img


"""
Retrieves the color flags associated with img

Params:
        * color - COLOR CODE

Returns: 
        * CV Color Code
"""
def getColorFlag(color):
        if color == 0: # MONO
                return 0
        elif color == 1: # BAYER
                return -1
        elif color == 2: # AS IS RBG
                return 1
                

"""
Backprojection - Finds objects of interest in an image

Params:
	* roi - Region of Interest Img
	* target - target img

Returns:
	* Img of Result
"""
def backProjection(roi, target):
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
	roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
	cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
	dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	cv2.filter2D(dst, -1, disc, dst)
	ret, thresh = cv2.threshold(dst, 50, 255, 0)
	thresh = cv2.merge((thresh, thresh, thresh))
	res = cv2.bitwise_and(target, thresh)
	return res


"""
Template Matching - Single Object, will find the points to draw a rectangle over given template

Params:
	* img - image
	* template - template image to match

Returns:
	* (top left, bottom right) - points of a rectangle
"""
def templateMatchSingle(img, template):
	img = grayscale(img)
	template = grayscale(template)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	return top_left, bottom_right


"""
Draws a rectangle over a matching pattern in img from template

Params:
	* img - image
	* template - pattern to search for
	* color - OPTIONAL - color; def: (255,255,0)
	* thickness - OPTIONAL - thickness of rectangle; def: 2

Returns:
	* img with rectangle over object
"""
def drawMatch(img, template, color=(255,255,0), thickness=2):
	tmp = img.copy()
	tl, br = templateMatchSingle(tmp, template)
	cv2.rectangle(tmp, tl, br, color, thickness)
	return tmp


"""
Saves an image to a file

Params:
	* img - image to save
	* filename - OPTIONAL - file to save to, def is TYEARMONTHDAY.jpg
"""
def save(img, filename=None):
	if filename is None:
		date = time.strftime("%Y%m%d")
		filename = "T" + str(date) + ".jpg"
		cv2.imwrite(filename, img)
	else:
		cv2.imwrite(filename, img)


"""
Returns Supported File Formats for Reading Images
"""
def getSupportedFileFormats():
        return {"Bitmap":["*.bmp", "*.dib"], "JPEG": ["*.jpeg", "*.jpg", "*.jpe"], "JPEG 2000": ["*.jp2"],"Portable Network Graphics" : ["*.png"], "WebP": ["*.webp"], "Portable Image Formats":["*.pbm", "*.pgm", "*.ppm"], "Sun Rasters":["*.sr", "*.ras"], "TIFF Files": ["*.tiff","*.tif"] }



"""
Saves a list of images to a file

Params:
	* img - images to save
	* filename - OPTIONAL - file names to save to, def is TYEARMONTHDAYITEM#.jpg
"""
def saveImgs(img, filename=None):
	if filename is None:
		date = time.strftime("%Y%m%d")
		filename = "T" + str(date)
		jpg = ".jpg"
		count = 0
		for item in img:
			name = filename + str(count) + jpg
			cv2.imwrite(name, item)
			count += 1
	else:
		for i in range(0, len(img)):
			cv2.imwrite(filename[i], img[i])


"""
Template Matching - Multiple Objects

Params:
	* img - image
	* template - image to search for

Returns:
	* lst of points in [(tl, br), (tl+w, br+h)] format for rectangles
"""
def templateMatchMulti(img, template):
	gray = grayscale(img)
	temp = grayscale(template)
	w, h = temp.shape[::-1]
	res = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where(res >= threshold)
	pts = []
	for pt in zip(*loc[::-1]):
		rect = [pt, (pt[0] + w, pt[1] + h)]
		pts.append(rect)
	return pts


"""
Draws a rectangle over each instance of an object it finds

Params:
	* img - image
	* template - template to search for
	* color - OPTIONAL - def (0,0,255)
	* thickness - OPTIONAL - def 2
"""
def drawMatchMulti(img, template, color = (0,0,255), thickness = 2):
	tmp = img.copy()
	gray = grayscale(img)
	temp = grayscale(template)
	w, h = temp.shape[::-1]
	res = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where(res >= threshold)
	for pt in zip(*loc[::-1]):
		cv2.rectangle(tmp, pt, (pt[0] + w, pt[1] + h), color, thickness)
	return tmp


"""
Resize an Image

Params:
	* img - image
	* width - new width
	* height - new height

Returns:
	* resized image
"""
def resize(img, width, height):
	tmp = img.copy()
	res = cv2.resize(tmp, (width, height), interpolation = cv2.INTER_CUBIC)
	return res


"""
Translates a image by shift (x, y)

Params:
	* img - image
	* shift - (x, y) tuple that has the shift direction in the x and y directions

Returns:
	* Translated image
"""
def translate(img, shift):
	gray = grayscale(img)
	tmp = img.copy()
	rows, cols = gray.shape
	M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]) # Translation Matrix
	dst = cv2.warpAffine(tmp, M, (cols, rows))
	return dst


"""
Affine Transformation - translates a list of points to a new set of points while keeping 
all parallel lines in original img parallel in output

Params:
	* img - image
	* pts - points of reference in original image
	* newPts - points to be translated to

Returns:
	* Shifted img under an Affine Transformation
"""
def affineTransform(img, pts, newPts):
	tmp = img.copy()
	if len(img.shape) is 3:
		rows, cols, ch = img.shape
	else:
		rows, cols = img.shape
	pts1 = np.float32(pts)
	pts2 = np.float32(newPts)
	M = cv2.getAffineTransform(pts1, pts2)
	dst = cv2.warpAffine(tmp, M, (cols, rows))
	return dst


"""
Perspective Transformation - Translates pts to newPts while keeping all lines straight

Params:
	* img - image
	* pts - 4 points on input image
	* newPts - 4 points to be shifted to
	* size - OPTIONAL - new size of output image (x,y), def is max newPts

Returns:
	Shifted img under Perspective Transformation
"""
def perspectiveTransform(img, pts, newPts, size=None):
	args = len(img.shape)
	tmp = img.copy()
	if args is 3:
		rows, cols, ch = img.shape
	else:
		rows, cols = img.shape
	pts1 = np.float32(pts)
	pts2 = np.float32(newPts)
	if size is None:
		xy = zip(*pts)
		pt = map(max, xy)
		size =(pt[0], pt[1])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(tmp, M, size)
	return dst


"""
Displays a list of images

Parms:
	* imgs - list of images
	* titles - OPTIONAL - list of titles of images
        * wait - OPTIONAL - wait time in ms ffor screen; def: 0 - INDEFINITE
"""
def displayImgs(imgs, titles = None, wait=0):
	if len(imgs) > 100:
		print "WARNING: DisplayImgs: List is of length " + str(len(imgs))
		print "Please reduce list size to avoid improper display"
		return
	if titles is None:
		count = 1
		for i in imgs:
			cv2.namedWindow("IMAGE" + str(count), cv2.WINDOW_NORMAL)
			cv2.imshow("IMAGE" + str(count), i)
			count += 1
	else:
		count = 0
		for i in imgs:
			cv2.namedWindow(titles[count], cv2.WINDOW_NORMAL)
			cv2.imshow(titles[count], i)
			count += 1
	cv2.waitKey(wait) & 0xFF
	cv2.destroyAllWindows()


"""
2D Convolution - Image Filtering

Params:
	* img - image to be filtered
	* kernel - OPTIONAL - size of average filtering kernel, def is (5,5)

Returns:
	* Filtered Image
"""
def filter2D(img, kernel = (5,5)):
	tmp = img.copy()
	k = np.ones((kernel[0], kernel[1]), np.float32) / (kernel[0]*kernel[1])
	dst = cv2.filter2D(tmp, -1, k)
	return dst


"""
Image Blur

Params:
	* img - image
	* kernel - OPTIONAL - size of kernel, def is (5,5)

Returns:
	* Blurred Image
"""
def blur(img, kernel = (5,5)):
	tmp = img.copy()
	blur = cv2.blur(tmp, kernel)
	return blur


"""
Gaussian Filtering/Blur

Params:
	* img - image
	* kernel - OPTIONAL - kernel size, def is (5,5)

Returns:
	* Gaussian Blurred Image
"""
def gaussianBlur(img, kernel = (5,5)):
	tmp = img.copy()
	blur = cv2.GaussianBlur(tmp, kernel, 0)
	return blur
	

"""
Median Filtering

Params:
	* img - Image

Returns:
	* Median Filtered Img
"""
def medianBlur(img):
	tmp = img.copy()
	return cv2.medianBlur(tmp, 5)


"""
Bilateral Filtering

Params: 
	* img - Image
	* d - OPTIONAL - diameter of each pixel neighborhood. def: 9
	* sigmaColor - OPTIONAL - Filter sigma in color space. def: 75
	* sigmaSpace - OPTIONAL - Filter sigma in coord space. def: 75

Returns:
	* Bilateral Filtered Img
"""
def bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75):
	tmp = img.copy()
	return cv2.bilateralFilter(tmp, d, sigmaColor, sigmaSpace)


"""
Adaptive Bilateral Filtering

Params:
	* img - image
	* ksize - OPTIONAL - Kernal Size, def: (5,5)
	* sigmaSpace - OPTIONAL - Filter sigma in coord space. def: None
"""
def adaptiveBilateralFilter(img, ksize=(5,5), sigmaSpace=None):
	if sigmaSpace is None:
		return cv2.adaptiveBilateralFilter(img, ksize)
	else:
		return cv2.adaptiveBilateralFilter(img, ksize, sigmaSpace=sigmaSpace)


"""
Erosion - Discard boundary pixels depending on kernel size

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)
	* iterations - OPTIONAL - number of iterations for erosion, def is 1

Returns:
	* eroded img
"""
def erode(img, kernel = (5,5), iterations = 1):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	erosion = cv2.erode(tmp, k, iterations= iterations)
	return erosion


"""
Dilation - Increases white region, size of foreground increases

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)
	* iterations - OPTIONAL - number of iterations for dilate, def is 1

Returns:
	* dilated img
"""
def dilate(img, kernel = (5,5), iterations = 1):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	dilation = cv2.dilate(tmp, k, iterations = iterations)
	return dilation


"""
Opening - Erosion followed by dilation. Useful for removing noise

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)

Returns:
	* opened img
"""
def opening(img, kernel = (5,5)):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	return cv2.morphologyEx(tmp, cv2.MORPH_OPEN, k)


"""
Closing - Dilation followed by Erosion. 
Useful for closing small holes inside foreground objects, 
or small black points on object

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)

Returns:
	* closed img
"""
def closing(img, kernel = (5,5)):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	return cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, k)


"""
Morphological Gradient - Difference btw Dilation and Erosion of img.
Usually results in outline of object

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)

Returns:
	* Gradient Img
"""
def gradient(img, kernel = (5,5)):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	return cv2.morphologyEx(tmp, cv2.MORPH_GRADIENT, k)


"""
Top Hat - Difference btw input img and Opening of img

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)

Returns:
	* TopHat Img
"""
def tophat(img, kernel = (5,5)):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	return cv2.morphologyEx(tmp, cv2.MORPH_TOPHAT, k)


"""
Black Hat - Difference btw closing of img and input img

Params:
	* img - image
	* kernel - OPTIONAL - kernel size in tuple, def is (5,5)

Returns:
	* BlackHat Img
"""
def blackhat(img, kernel = (5,5)):
	tmp = grayscale(img)
	k = np.ones(kernel, np.uint8)
	return cv2.morphologyEx(tmp, cv2.MORPH_BLACKHAT, k)


"""
Hit And Miss Morphological Transform

Params:
	* img - Image

Returns:
	* HitAndMiss Img
"""
def hitAndMiss(img):
	tmp = img.copy()
	return cv2.morphologyEx(tmp, cv2.MORPH_HITMISS)


"""
Get a rectangular kernel

Params:
	* size - tuple of size of requested kernel, def is (5,5)

Returns:
	* desired kernel
"""
def getRectangularKernel(size = (5,5)):
	return cv2.getStructuringElement(cv2.MORPH_RECT, size)


"""
Get a elliptical kernel

Params:
	* size - tuple of size of requested kernel, def is (5,5)

Returns:
	* desired kernel
"""
def getEllipticalKernel(size = (5,5)):
	return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)


"""
Get a cross-shaped kernel

Params:
	* size - tuple of size of requested kernel, def is (5,5)

Returns:
	* desired kernel
"""
def getCrossKernel(size = (5,5)):
	return cv2.getStructuringElement(cv2.MORPH_CROSS, size)


"""
Laplacian Image Gradient

Params:
	* img -image

Returns:
	* Laplacian Image
"""
def laplacian(img):
	gray = grayscale(img)
	return cv2.Laplacian(gray, cv2.CV_64F)


"""
Sobel X Image Gradient

Params:
	* img -image
	* ksize - OPTIONAL - Kernel Size, def is 5

Returns:
	* Sobel X Image
"""
def sobelx(img, ksize=5):
	gray = grayscale(img)
	return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= ksize)


"""
Sobel Y Image Gradient

Params:
	* img -image
	* ksize - OPTIONAL - Kernel Size, def is 5

Returns:
	* Sobel Y Image
"""
def sobely(img, ksize=5):
	gray = grayscale(img)
	return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= ksize)


"""
Sobel XY Image Gradient

Params:
	* img -image
	* ksize - OPTIONAL - Kernel Size, def is 5

Returns:
	* Sobel XY Image
"""
def sobelxy(img, ksize=5):
	gray = grayscale(img)
	return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize= ksize)


"""
Draws the Found Lines onto the Image
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* rho- OPTIONAL - radius, measured in pixels, def is 1
	* theta - OPTIONAL - angle, measured in radians, def is np.pi/180
	* color - OPTIONAL - def (0,0,255)
	* threshold - OPTIONAL - Minimum Length of Line, def 200
	* thickness - OPTIONAL - def 2

Returns:
	* Image with lines plotted
"""
def drawHoughLines(img, rho=1, theta=np.pi/180, color=(0,0,255), threshold=200, thickness=2):
	tmp = img.copy()
	gray = grayscale(img)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, rho, theta, threshold)
	if lines is None:
		print "No lines found, please adjust params...\n"
		return None
	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0=a*rho
		y0=b*rho
		x1 = int(x0 + img.shape[1]*(-b))
		y1 = int(y0 + img.shape[1]*(a))
		x2 = int(x0 - img.shape[1]*(-b))
		y2 = int(y0 - img.shape[1]*(a))
		cv2.line(tmp, (x1, y1), (x2,y2), color, thickness)
	return tmp


"""
Hough Lines Detection Method
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* rho- OPTIONAL - radius, measured in pixels, def is 1
	* theta - OPTIONAL - angle, measured in radians, def is np.pi/180
	* threshold - OPTIONAL - Minimum Length of Line, def 200
Returns:
	* List of points in [(start, end)] format
"""
def houghLines(img, rho=1, theta=np.pi/180, threshold=200):
	tmp = img.copy()
	gray = grayscale(img)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, rho, theta, threshold)
	if lines is None:
		print "No lines found, please adjust params...\n"
		return None
	pts = []
	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0=a*rho
		y0=b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		pts.append([(x1,y1), (x2,y2)])
	return pts


"""
Draws the result of Probabilistic Hough Line Transform onto img
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* minLineLength - OPTIONAL - Min length of line, any lines shorter rejected, def is 100
	* maxLineGap - OPTIONAL - Max allowed gap btw line segments to treat as single line, def is 10
	* color - OPTIONAL - color of line, def is (0,255,0)
	* thickness - OPTIONAL - def is 2

Returns:
	* img with lines draw onto it
""" 
def drawHoughLinesProb(img, minLineLength=100, maxLineGap = 10, color=(0,255,0), thickness=2):
	gray = grayscale(img)
	tmp = img.copy()
	edges = cv2.Canny(gray, 50, 150, apetureSize = 3)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
	if lines is None:
		print "No lines found, please adjust params...\n"
		return None
	for x1, y1, x2, y2 in lines[0]:
		cv2.line(tmp, (x1,y1), (x2,y2), color, thickness)
	return tmp


"""
Returns the result of Probabilistic Hough Line Transform from img
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* minLineLength - OPTIONAL - Min length of line, any lines shorter rejected, def is 100
	* maxLineGap - OPTIONAL - Max allowed gap btw line segments to treat as single line, def is 10

Returns:
	* List of lines in [(start, end)] format
""" 
def houghLinesProb(img, minLineLength=100, maxLineGap = 10):
	gray = grayscale(img)
	tmp = img.copy()
	edges = cv2.Canny(gray, 50, 150, apetureSize = 3)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
	if lines is None:
		print "No lines found, please adjust params...\n"
		return None
	pts = []
	for x1, y1, x2, y2 in lines[0]:
		pts.append([(x1,y1), (x2,y2)])
	return pts


"""
Hough Circle Transform - Finds Circles in an image and draws it
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* minDist - OPTIONAL - Minimum Distance btw centers of detected circles, def is 20
	* param1 - OPTIONAL - Higher Threshold of two sent to Canny edge detector, def is 50
	* param2 - OPTIONAL - Accumulator Threshold for the circle centers at detection stage, def is 30
	* minRadius - OPTIONAL - Minimum Circle Radius, def is 0
	* maxRadius - OPTIONAL - Maximum Circle Radius, def is 0
	* colorCircle - OPTIONAL - color of circle, def is (0,255,0)
	* colorCenter - OPTIONAL - color of circle's center, def is (0,0,255)
	* centerRadius - OPTIONAL - radius of center
	* thickness - OPTIONAL - thickness of circles

Returns:
	* Image with circles and their centers drawn
"""
def drawHoughCircles(img, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0 ,colorCircle=(0,255,0), colorCenter=(0,0,255), centerRadius=2 , thickness=2):
	tmp = grayscale(img)
	tmp = cv2.medianBlur(tmp, 5)
	cimg = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(tmp, cv.CV_HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
	if circles is None:
		print "No circles found, please adjust params...\n"
		return None
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		cv2.circle(cimg, (i[0],i[1]),i[2], colorCircle, thickness)
		cv2.circle(cimg, (i[0],i[1]), centerRadius, colorCenter, thickness)
	return cimg


"""
Hough Circle Transform - Finds Circles in an image and draws it
NOTE: Play around with params to get what you need, each img has different req params

Params:
	* img - image
	* minDist - OPTIONAL - Minimum Distance btw centers of detected circles, def is 20
	* param1 - OPTIONAL - Higher Threshold of two sent to Canny edge detector, def is 50
	* param2 - OPTIONAL - Accumulator Threshold for the circle centers at detection stage, def is 30
	* minRadius - OPTIONAL - Minimum Circle Radius, def is 0
	* maxRadius - OPTIONAL - Maximum Circle Radius, def is 0

Returns:
	* List of circles in (x, y, radius) format
"""
def houghCircles(img, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
	tmp = grayscale(img)
	tmp = cv2.medianBlur(tmp, 5)
	cimg = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(tmp, cv.CV_HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
	if circles is None:
		print "No circles found, please adjust params...\n"
		return None
	circles = np.uint16(np.around(circles))
	return circles


"""
ORB Corner Detection - Detects Corners

Params:
	* img - image

Returns:
	* key points
"""
def orb(img):
	gray = grayscale(img)
	orb = cv2.ORB()
	kp = orb.detect(gray, None)
	kp, des = orb.compute(img, kp)
	return kp


"""
ORB Corner Detection

Params:
	* img - image

Returns:
	* img with key points drawn on
	* color- OPTIONAL - color, def is (0,255,0)
"""
def drawOrb(img, color = (0,255,0)):
	gray = grayscale(img)
	orb = cv2.ORB()
	kp = orb.detect(gray, None)
	kp, des = orb.compute(img, kp)
	img2 = cv2.drawKeypoints(gray, kp, color=color, flags=0)
	return img2


"""
Harris Corner Detection

Params:
	* img - image
	* blockSize - OPTIONAL - size of neighborhood considered for corner detection
	* ksize - OPTIONAL - Aperture parameter of Sobel derivative used
	* k - OPTIONAL - Harris detector free parameter in equation
	* color - OPTIONAL - def is (0,0,255) 

Returns:
	* Image with corners marked.
"""
def harrisCorner(img, blockSize=2, ksize=3, k=0.04, color=(0,0,255)):
	tmp = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,blockSize, ksize, k)
	dst = cv2.dilate(dst, None)
	tmp[dst>0.01*dst.max()] = color
	return tmp


"""
Harris Corner Detection with SubPixel Accuracy

Params:
	* img - image
	* blockSize - OPTIONAL - size of neighborhood considered for corner detection
	* ksize - OPTIONAL - Aperture parameter of Sobel derivative used
	* k - OPTIONAL - Harris detector free parameter in equation

Returns:
	* Corners.	
"""
def harrisSubPixel(img, blockSize=2, ksize=3, k=0.04):
	tmp = img.copy()
	gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, blockSize, ksize, k)
	dst = cv2.dilate(dst, None)
	ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
	return corners


"""
Good Features to Track Corner Detection Shi Tomasi

Params:
	* img - image
	* numCorners - OPTIONAL - Number of Corners, def is 25
	* quality - OPTIONAL - min quality of corner, btw 0-1, def is 0.01
	* minDist - OPTIONAL - Min Euclidean Dist btw Corners Detected
	* radius - OPTIONAL - radius, def is 3
	* color - OPTIONAL - def is (0,0,255)

Returns:
	* img with corners marked
"""
def goodFeaturesToTrack(img, numCorners=25, quality=0.01, minDist=10, radius=3, color=(0,0,255)):
	tmp = img.copy()
	gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray, numCorners, quality, minDist)
	corners = np.int0(corners)
	for i in corners:
		x,y = i.ravel()
		cv2.circle(tmp, (x,y), radius, color, -1)
	return tmp


"""
Good Features to Track Corner Detection Shi Tomasi

Params:
	* img - image
	* numCorners - OPTIONAL - Number of Corners, def is 25
	* quality - OPTIONAL - min quality of corner, btw 0-1, def is 0.01
	* minDist - OPTIONAL - Min Euclidean Dist btw Corners Detected
	* radius - OPTIONAL - radius, def is 3
	* color - OPTIONAL - def is (0,0,255)

Returns:
	* corners detected
"""
def goodFeaturesToTrackPts(img, numCorners=25, quality=0.01, minDist=10, radius=3, color=(0,0,255)):
	tmp = img.copy()
	gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray, numCorners, quality, minDist)
	corners = np.int0(corners)
	return corners


"""
FAST Algorithm for Corner Detection - Returns Key Points

Params:
	* img - image
	* nonmaxSuppression - OPTIONAL - Suppresses number of points if True, def is True

Returns:
	* Key Points
"""
def fast(img, nonmaxSuppression = True):
	gray = grayscale(img)
	fast = cv2.FastFeatureDetector()
	if not nonmaxSuppression:
		fast.setBool('nonmaxSuppression', 0)
	kp = fast.detect(img, None)
	return kp
	

"""
FAST Algorithm for Corner Detection - Returns Key Points

Params:
	* img - image
	* nonmaxSuppression - OPTIONAL - Suppresses number of points if True, def is True
	* color - OPTIONAL - def: (255,0,0)

Returns:
	* Key Points drawn onto img
"""	 
def drawFast(img, nonmaxSuppression=True, color=(255,0,0)):
	tmp = img.copy()
	kp = fast(tmp, nonmaxSuppression=nonmaxSuppression)
	tmp = cv2.drawKeypoints(tmp, kp, color=color)
	return tmp


"""
Draw Key Points

Params:
	* img - image
	* kp - Key Pints List
	* color - OPTIONAL - def: (255,0,0)

Returns:
	* img with keypoints
"""
def drawKeyPoints(img, kp, color=(255,0,0)):
	tmp = img.copy()
	tmp = cv2.drawKeypoints(tmp, kp, color = color)
	return tmp


"""
Image De-noising - Both Colored and Grayscale imgs

Params:
	* img - image
	* h - OPTIONAL - filter strength; def: 10
	* hForColor - OPTIONAL - used if img is color; same as h; def: h
	* templateWindowSize - OPTIONAL - odd num; def: 7
	* searchWindowSize - OPTIONAL - odd num; def: 21

Returns:
	* De-noised img
"""
def denoise(img, h=10, hForColor=None, templateWindowSize=7, searchWindowSize=21):
	if hForColor is None:
		hForColor=h
	tmp = img.copy()
	if len(img.shape) != 3:
		dst = cv2.fastNlMeansDenoising(tmp, None, h, templateWindowSize, searchWindowSize)
	else:
		dst = cv2.fastNlMeansDenoisingColored(img, None, h, hForColor, templateWindowSize, searchWindowSize)
	return dst	


"""
Brute-Force Matcher - Matches

Params:
	* img1 - Query Image 
	* img2 - Train Image

Returns:
	* Matches
"""
def BFMatch(img1, img2):
	gray1 = grayscale(img1)
	gray2 = grayscale(img2)
	orb = cv2.ORB()
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key = lambda x:x.distance)
	return matches


"""
Brute-Force Matcher - Draw Img

Params:
	* img1 - Query Image 
	* img2 - Train Image
	* numMatches - OPTIONAL - Number of Matches to Display; def: All

Returns:
	* Img with matches drawn
"""
def drawBFMatch(img1, img2, numMatches=None):
	matches = BFMatch(img1, img2)
	if numMatches is None:
		img3 = cv2.drawMatches(img1, kp1m, img2, kp2, matches, flags=2)
	else:
		img3 = cv2.drawMatches(img1, kp1m, img2, kp2, matches[:numMatches], flags=2)
	return img3


"""
Slope Calculator from two points

Params:
	* start - start point (x,y)
	* end - end point (x,y)

Returns:
	* slope of line connecting two pts, None if undefined
"""
def slope(start, end):
	x1 = start[0]
	y1 = start[1]
	x2 = end[0]
	y2 = end[1]
	top = float(y2 - y1) 
	bot = float(x2 - x1)
	if bot == 0:
		return None
	else:
		return top / bot


"""
Distance Between Two Points

Params:
	* pt1 - first point, (x,y)
	* pt2 - second point, (x,y)

Returns:
	* Distance Between Two Points
"""
def distance(pt1, pt2):
	x1, y1 = pt1
	x2, y2 = pt2
	x = x2 - x1
	y = y2 - y1
	s = x**2 + y**2
	return np.sqrt(s)


"""
Returns a List of all Events handled by OpenCV

Params:
	* filter - OPTIONAL - filter list results
Returns:
	* list of events
"""
def eventList(filterStr=""):
	filterStr = filterStr.upper()
	events = [i for i in dir(cv2) if 'EVENT' in i and filterStr in i]
	return events


"""
Adds a border to the image

Params:
	* img - image to add border
	* flag - OPTIONAL - border type flag, def is CONSTANT (0)
	* top - OPTIONAL - top pixel width
	* bottom - OPTIONAL - bottom pixel width
	* right - OPTIONAL - right pixel width
	* left - OPTIONAL - left pixel width
	* color - OPTIONAL - only used if flag == cv2.BORDER_CONSTANT

Returns:
	* img with specified border
"""
def addBorder(img, flag=0, top=10, bottom=10, left=10, right=10, color = (255,0,0)):
	if flag != 0:
		borderImg = cv2.copyMakeBorder(img, top, bottom, left, right, flag)
		return borderImg
	elif flag == 0:
		borderImg = cv2.copyMakeBorder(img, top, bottom, left, right, flag, value=color)
		return borderImg
	else:
		print "ERROR: AddBorder: Invalid Flag"
		sys.exit()


"""
Returns border flags and int values

Returns:
	* Border values
"""
def getBorderFlags():
	return border_flag


"""
Integral of an Img

Params:
	* img - image
	* sqSum - OPTIONAL - if True, returns integral for squared pixel values
	* tilted - OPTIONAL - if True, returns integral for img rotated by 45 deg

Returns:
	* integral
	* sq pixel integral
	* tilted integral
"""
def integral(img, sqSum = False, tilted = False):
	if sqSum is False and tilted is False:
		return cv2.integral(img)
	elif sqSum is True and tilted is False:
		return cv2.integral2(img)
	elif sqSum is True and tilted is True:
		return cv2.integral3(img)
	elif sqSum is False and tilted is True:
		su, sqsu, tilt = cv2.integral3(img)
		return su, tilt
	else:
		return cv2.integral(img)


#### END OF IMAGE METHODS ####
#### START OF VIDEO METHODS - EXPERIMENTAL ####


"""
Capture and display an image from Camera

Params:
	* title - OPTIONAL - name of display pop up, def: 'Frame'
"""
def captureDisplay(title="Frame"):
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.imshow(title, frame)
	cap.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()


"""
Captures and returns an image from Camera

Returns:
	* Captured Img
"""
def capture():
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	cap.release()
	cv2.destroyAllWindows()
	return frame


"""
Object Tracking Video

Params:
	* lower - Lower Bound
	* upper - Upper Bound
"""
def objectTrackVid(lower, upper):
	cap = cv2.VideoCapture(0)
	print "Please hit ESC key when done to ensure camera closure..."
	while(1):
		_, frame = cap.read()
		res = trackObject(frame, lower, upper)
		cv2.imshow('Frame', frame)
		cv2.imshow('Result', res)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

# End of File
