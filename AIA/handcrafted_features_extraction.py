import cv2
import skimage
import numpy as np
import math
import csv
import sys
from tqdm import tqdm

def triangleAutoThreshold(img):
    # number of gray levels
    grayLevels  = 256
    
    histo = cv2.calcHist(img, [1], None, [grayLevels], (0, 256), False)
    
    # find min and max
    min = 0
    dmax = 0
    max = 0
    min2 = 0
    for i in range(len(histo)):
        if histo[i] > 0:
            min = i
            break
        
    if min > 0:
        min -= 1 # line to the (p==0) point, not to histo[min]
    
    for i in range(len(histo) - 1, 0, -1):
        if histo[i] > 0:
            min2 = i
            break
    
    if min2 < len(histo) - 1:
        min2 += 1 # line to the (p==0) point, not to histo[min]
    
    for i in range(len(histo)):
        if histo[i] > dmax:
            max = i
            dmax = histo[i]
            
    # find which is the furthest side
    inverted = False
    if (max - min) < (min2 - max):
        # reverse the histogram
        inverted = True
        left = 0                #index of leftmost element
        right = len(histo) - 1 # index of rightmost element
        while (left < right):
            # exchange the left and right elements
            temp = histo[left]
            histo[left]  = histo[right]
            histo[right] = temp
            # move the bounds toward the center
            left += 1
            right -= 1
        min = len(histo) - 1 - min2
        max = len(histo) - 1 - max
    
    if min == max:
        return min
    
    # describe line by nx * x + ny * y - d = 0
    # nx is just the max frequency as the other point has freq=0
    nx = histo[max]   #-min; # histo[min]; #  lowest value bmin = (p=0)% in the image
    ny = min - max
    d = math.sqrt(nx * nx + ny * ny)
    nx /= d
    ny /= d
    d = nx * min + ny * histo[min]
    
    # find split point
    split = min
    splitDistance = 0
    for i in range(min + 1, max + 1):
        newDistance = nx * i + ny * histo[i] - d
        if newDistance > splitDistance:
            split = i
            splitDistance = newDistance
    split -= 1
    
    if inverted:
        # The histogram might be used for something else, so let's reverse it back
        left  = 0; 
        right = len(histo) - 1
        while left < right:
            temp = histo[left] 
            histo[left]  = histo[right]
            histo[right] = temp
            left += 1
            right -= 1
        return len(histo) - 1 - split
    else:
        return split



projectPath = "./"
imagesPath = projectPath + "images/"
featuresPath = projectPath + "features/"
datasetFileName = projectPath + "Galaxy10_DECals.csv"

datasetFile = open(datasetFileName, 'r')
featuresFile = open(featuresPath + "features.csv", 'w')
geometricFile = open(featuresPath + "geometric_features.csv", 'w')
GLCMFile = open(featuresPath + "glcm_features.csv", 'w')
LBPFile = open(featuresPath + "lbp_features.csv", 'w')
gaborFile = open(featuresPath + "gabor_features.csv", 'w')
fourierFile = open(featuresPath + "fourier_features.csv", 'w')

csvReader = csv.reader(datasetFile, delimiter=',')
csvWriterAll = csv.writer(featuresFile)
csvWriterGeometric = csv.writer(geometricFile)
csvWriterGLCM = csv.writer(GLCMFile)
csvWriterLBP = csv.writer(LBPFile)
csvWriterGabor = csv.writer(gaborFile)
csvWriterFourier = csv.writer(fourierFile)


# features' file header
featuresHeader = []

# geometric and statistical features
geometricHeader = []
geometricHeader.append("FILENAME")

geometricHeader.append("AREA")
geometricHeader.append("PERIMETER")
geometricHeader.append("HU1")
geometricHeader.append("HU2")
geometricHeader.append("HU3")
geometricHeader.append("HU4")
geometricHeader.append("HU5")
geometricHeader.append("HU6")
geometricHeader.append("HU7")
geometricHeader.append("AXIS_RATIO")
geometricHeader.append("CIRCULARITY")
geometricHeader.append("ECCENTRICITY")
geometricHeader.append("UNIFORMITY")
geometricHeader.append("ENTROPY")

geometricHeader.append("CLASS")

# GLCM features
GLCMHeader = []
GLCMHeader.append("FILENAME")

GLCMAngles = np.arange(0, np.pi, np.pi / 4)
for i in range(len(GLCMAngles)):
    GLCMHeader.append("CONTRAST" + str(i))
    GLCMHeader.append("DISSIMILARITY" + str(i))
    GLCMHeader.append("HOMOGENEITY" + str(i))
    GLCMHeader.append("ENERGY" + str(i))
    GLCMHeader.append("CORRELATION" + str(i))
    GLCMHeader.append("ASM" + str(i))

GLCMHeader.append("CLASS")

# LBP features
LBPHeader = []
LBPHeader.append("FILENAME")

for i in range(3):  # 3 histograms combined
    for j in range(256):    # 256 bins
        LBPHeader.append("HIST" + str(i) + "_" + str(j))

LBPHeader.append("CLASS")

# gabor features
thetas = np.arange(0, np.pi, np.pi / 4)
lambdas = np.arange(np.pi / 4, np.pi, np.pi / 4)
sigmas = np.arange(1, 6, 2)
gaborNumber = len(thetas) * len(lambdas)* len(sigmas)

gaborHeader = []
gaborHeader.append("FILENAME")

for i in range(gaborNumber):
    gaborHeader.append("MEAN" + str(i))
    gaborHeader.append("STD_DEV" + str(i))

gaborHeader.append("CLASS")

# fourier descriptors
descriptorsNumber = 10

fourierHeader = []
fourierHeader.append("FILENAME")

for i in range(descriptorsNumber):
    fourierHeader.append("FOURIER_REAL" + str(i))
    fourierHeader.append("FOURIER_IMAG" + str(i))

fourierHeader.append("CLASS")

featuresHeader = geometricHeader[0:len(geometricHeader) - 1] + GLCMHeader[1:len(GLCMHeader) - 1] + LBPHeader[1:len(LBPHeader) - 1] + gaborHeader[1:len(gaborHeader) - 1] + fourierHeader[1:len(fourierHeader)]

featuresVector = [None] * len(featuresHeader)
geometricFeaturesVector = [None] * len(geometricHeader)
GLCMFeaturesVector = [None] * len(GLCMHeader)
LBPFeaturesVector = [None] * len(LBPHeader)
gaborFeaturesVector = [None] * len(gaborHeader)
fourierFeaturesVector = [None] * len(fourierHeader)

csvWriterAll.writerow(featuresHeader)
csvWriterGeometric.writerow(geometricHeader)
csvWriterGLCM.writerow(GLCMHeader)
csvWriterLBP.writerow(LBPHeader)
csvWriterGabor.writerow(gaborHeader)
csvWriterFourier.writerow(fourierHeader)

for row in tqdm(csvReader):
    # filename
    geometricFeaturesVector[0] = GLCMFeaturesVector[0] = LBPFeaturesVector[0] = gaborFeaturesVector[0] = fourierFeaturesVector[0] = row[0]  
    
    # class
    geometricFeaturesVector[len(geometricFeaturesVector) - 1] = row[1]
    GLCMFeaturesVector[len(GLCMFeaturesVector) - 1] = row[1]
    LBPFeaturesVector[len(LBPFeaturesVector) - 1] = row[1]
    gaborFeaturesVector[len(gaborFeaturesVector) - 1] = row[1]
    fourierFeaturesVector[len(fourierFeaturesVector) - 1] = row[1]

    # image reading
    imgName = imagesPath + row[0] + ".png"
    original = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Image", original)
    # cv2.waitKey(0)

    # Non-Local Means Filtering
    denoised = cv2.fastNlMeansDenoising(original, None, 10, 7, 21)
    # cv2.imshow("NLM", denoised)
    # cv2.waitKey(0)
    
    galaxy_image = denoised.copy()

    # # thresholding with triangle method
    # _, galaxy_image = cv2.threshold(galaxy_image, triangleAutoThreshold(original), 255, cv2.THRESH_BINARY)
    # cv2.imshow("Thresholding", galaxy_image)
    # cv2.waitKey(0)

    # thresholding
    _, galaxy_image = cv2.threshold(galaxy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Thresholding", galaxy_image)
    # cv2.waitKey(0)
    
    # opening
    galaxy_image = cv2.morphologyEx(galaxy_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # cv2.imshow("Opening", galaxy_image)
    # cv2.waitKey(0)

    # canny
    galaxy_image = cv2.Canny(galaxy_image, 100, 200)
    # cv2.imshow("Canny", galaxy_image)
    # cv2.waitKey(0)

    # find contours
    contours, _ = cv2.findContours(galaxy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # geometric moments
    moments = [None] * len(contours)

    centroids = [None] * len(contours)
    distances = [None] * len(contours)

    for i in range(len(contours)):
        moments[i] = cv2.moments(contours[i])
        
        x = np.int32(moments[i]['m10']/(moments[i]['m00'] + sys.float_info.epsilon))
        y = np.int32(moments[i]['m01']/(moments[i]['m00'] + sys.float_info.epsilon))
        centroids[i] = [x, y]
        
        # square distance (it is only used to compare the distance of each object from the center of the image so the sqrt is not needed)
        distances[i] = (x - 128) ** 2 + (y - 128) ** 2

    # index of the object closer to the center of the image (galaxy)
    minDistIndex = np.argmin(distances)

    # galaxy's bounding rectangle
    boundingRect = cv2.boundingRect(contours[minDistIndex])
    x1 = np.int32(boundingRect[0])
    y1 = np.int32(boundingRect[1])
    x2 = np.int32(boundingRect[0] + boundingRect[2])
    y2 = np.int32(boundingRect[1] + boundingRect[3])

    # roi building from bounding rectangle
    roi = denoised[y1:y2, x1:x2]
    # cv2.imshow("ROI", roi) 
    # cv2.waitKey(0)

    # geometric features
    area = moments[minDistIndex]['m00']
    perimeter = cv2.arcLength(contours[minDistIndex], True)
    huMoments = cv2.HuMoments(moments[minDistIndex])
    circularity = 4 * math.pi * area / (perimeter ** 2)
    
    # inertia matrix for ellipsoid
    tr = moments[minDistIndex]['mu20'] + moments[minDistIndex]['mu02']
    det = moments[minDistIndex]['mu20'] * moments[minDistIndex]['mu02'] - moments[minDistIndex]['mu11'] * moments[minDistIndex]['mu11']

    # eigenvalues
    l1 = tr + math.sqrt((tr * tr - 4 * det) / 2)
    l2 = tr - math.sqrt((tr * tr - 4 * det) / 2)

    # ellipsoid's axis
    if l1 > l2:
        a = np.int32(math.sqrt(l1 / (moments[i]['m00'] + sys.float_info.epsilon)))
        b = np.int32(math.sqrt(l2 / (moments[i]['m00'] + sys.float_info.epsilon)))
    else:
        a = np.int32(math.sqrt(l2 / (moments[i]['m00'] + sys.float_info.epsilon)))
        b = np.int32(math.sqrt(l1 / (moments[i]['m00'] + sys.float_info.epsilon)))

    # ellipsoid's axis ratio
    axis_ratio = a / (b + sys.float_info.epsilon)

    # minimum area rectangle
    minAreaRect = cv2.minAreaRect(contours[minDistIndex])
    width = minAreaRect[1][0]
    height = minAreaRect[1][1]

    # eccentricity
    eccentricity = width / height

    # roi's normalized histogram
    (hist, _) = np.histogram(roi.ravel(), 256, (0, 255), True)

    # statistical features
    uniformity = 0
    entropy = 0
    for i in range(256):
        uniformity += hist[i] ** 2 
        entropy += hist[i] * np.log2(hist[i] + sys.float_info.epsilon)
    entropy *= -1

    geometricFeaturesVector[1] = area
    geometricFeaturesVector[2] = perimeter
    
    for j in range(len(huMoments)):
        geometricFeaturesVector[j + 3] = huMoments[j][0]
    
    geometricFeaturesVector[10] = axis_ratio
    geometricFeaturesVector[11] = circularity
    geometricFeaturesVector[12] = eccentricity
    geometricFeaturesVector[13] = uniformity
    geometricFeaturesVector[14] = entropy

    # print(geometricFeaturesVector)

    # GLCM matrix
    GLCMMatrix = skimage.feature.graycomatrix(roi, [1], GLCMAngles, levels = 256, normed = True)

    # GLCM features
    contrast = skimage.feature.graycoprops(GLCMMatrix, 'contrast')
    dissimilarity = skimage.feature.graycoprops(GLCMMatrix, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(GLCMMatrix, 'homogeneity')
    energy = skimage.feature.graycoprops(GLCMMatrix, 'energy')
    correlation = skimage.feature.graycoprops(GLCMMatrix, 'correlation')
    asm = skimage.feature.graycoprops(GLCMMatrix, 'ASM')
    
    for i in range(len(GLCMAngles)):
        GLCMFeaturesVector[6 * i + 1] = contrast[0][i]
        GLCMFeaturesVector[6 * i + 2] = dissimilarity[0][i]
        GLCMFeaturesVector[6 * i + 3] = homogeneity[0][i]
        GLCMFeaturesVector[6 * i + 4] = energy[0][i]
        GLCMFeaturesVector[6 * i + 5] = correlation[0][i]
        GLCMFeaturesVector[6 * i + 6] = asm[0][i]

    # print(GLCMFeaturesVector) 

    # LBP features
    # 1st LBP histogram
    LBP = skimage.feature.local_binary_pattern(roi, 8, 1, 'uniform')
    (hist, _) = np.histogram(LBP.ravel(), 256, (0, 255), True)
    for i in range(len(hist)):
        LBPFeaturesVector[i + 1] = hist[i]
    
    # 2nd LBP histogram with blurred roi with gaussian blur 13x13
    gaussian = cv2.GaussianBlur(roi, (13, 13), 0)
    LBP = skimage.feature.local_binary_pattern(roi, 8, 3, 'uniform')
    (hist, _) = np.histogram(LBP.ravel(), 256, (0, 255), True)
    for i in range(len(hist)):
        LBPFeaturesVector[256 + i + 1] = hist[i]

    # 3rd LBP histogram with blurred roi with gaussian blur 21x21
    gaussian = cv2.GaussianBlur(roi, (21, 21), 0)
    LBP = skimage.feature.local_binary_pattern(roi, 8, 5, 'uniform')
    (hist, _) = np.histogram(LBP.ravel(), 256, (0, 255), True)
    for i in range(len(hist)):
        LBPFeaturesVector[512 + i + 1] = hist[i]
    
    # print(LBPFeaturesVector)

    # gabor's feature

    mean = [None] * gaborNumber
    std_dev = [None] * gaborNumber
    i = 0
    for theta in thetas:
            for lambd in lambdas:
                for sigma in sigmas:
                    kernel = cv2.getGaborKernel((7, 7), sigma, theta, lambd, 0, 0, cv2.CV_32F)
                    kernel /= kernel.sum()
                    filtered_image = cv2.filter2D(roi, cv2.CV_8U, kernel)
                    filtered_image = filtered_image.reshape(-1)
                    mean[i] = np.mean(filtered_image)
                    std_dev[i] = np.std(filtered_image)
                    i += 1
    for i in range(gaborNumber):
        gaborFeaturesVector[2 * i + 1] = mean[i]
        gaborFeaturesVector[2 * i + 2] = std_dev[i]
    
    # print(gaborFeaturesVector)

    # fourier descriptors
    fourierDescriptors = cv2.ximgproc.contourSampling(contours[minDistIndex], descriptorsNumber)

    for i in range(descriptorsNumber):
        fourierFeaturesVector[2 * i + 1] = fourierDescriptors[i][0][0]
        fourierFeaturesVector[2 * i + 2] = fourierDescriptors[i][0][1]

    # print(fourierFeaturesVector)

    featuresVector = geometricFeaturesVector[0:len(geometricFeaturesVector) - 1] + GLCMFeaturesVector[1:len(GLCMFeaturesVector) - 1] + LBPFeaturesVector[1:len(LBPFeaturesVector) - 1] + gaborFeaturesVector[1:len(gaborFeaturesVector) - 1] + fourierFeaturesVector[1:len(fourierFeaturesVector)]
    
    csvWriterAll.writerow(featuresVector)
    csvWriterGeometric.writerow(geometricFeaturesVector)
    csvWriterGLCM.writerow(GLCMFeaturesVector)
    csvWriterLBP.writerow(LBPFeaturesVector)
    csvWriterGabor.writerow(gaborFeaturesVector)
    csvWriterFourier.writerow(fourierFeaturesVector)

datasetFile.close()
featuresFile.close()
geometricFile.close()
GLCMFile.close()
LBPFile.close()
gaborFile.close()
fourierFile.close()