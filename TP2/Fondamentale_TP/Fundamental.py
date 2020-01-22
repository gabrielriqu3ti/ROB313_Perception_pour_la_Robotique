import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


###### Read images
img1 = cv2.imread('POP01.jpg',0)  #queryimage # left image
img2 = cv2.imread('POP02.jpg',0) #trainimage # right image
#img1 = cv2.imread('DeathStar1.jpg',0)  #queryimage # left image
#img2 = cv2.imread('DeathStar2.jpg',0) #trainimage # right image

print(img1.shape)


###### Detect and match keypoints
kaze = cv2.KAZE_create(upright = False,#Par défaut : false
                      threshold = 0.003,#Par défaut : 0.001
                      nOctaves = 4,#Par défaut : 4
                      nOctaveLayers = 8,#Par défaut : 4
                      diffusivity = 2)#Par défaut : 2


# find the keypoints and descriptors with KAZE
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)

print('Nb of keypoints: ' + str(len(kp1)) + ' ' + str(len(kp2)))
imgd=img1
cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Keypoints', img1.shape[0], img1.shape[1])
imgd = cv2.drawKeypoints(img1, kp1, imgd,-1,flags=4)
cv2.imshow('Keypoints', imgd)
cv2.waitKey(0)
imgd=img2
imgd = cv2.drawKeypoints(img2, kp2, imgd,-1,flags=4)
cv2.imshow('Keypoints', imgd)
cv2.waitKey(0)
cv2.destroyAllWindows()


# match keypoints using FLANN library
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

m_image = np.array([])
m_image = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    [match[0] for match in matches],
    m_image)
cv2.namedWindow('Match', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Match', 2*img1.shape[0], img1.shape[1])
cv2.imshow('Match', m_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

pts1 = []
pts2 = []

# filter matching using threshold and ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if (m.distance < 0.2) & (m.distance < 0.99*n.distance):
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

print('Number of matched points : ' + str(pts1.shape[0]))


##### Definition of some helper functions
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color=tuple(cv2.cvtColor(np.asarray([[[np.random.randint(0,180),255,255]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0,:].tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def drawFundamental(img1,img2,pts1,pts2,F):
    # Find epilines corresponding to some points in right image (second image) and
    # drawing its lines on left image
    indexes = np.random.randint(0, pts1.shape[0], size=(10))
    indexes=range(pts1.shape[0])
    samplePt1 = pts1[indexes,:]
    samplePt2 = pts2[indexes,:]

    lines1 = cv2.computeCorrespondEpilines(samplePt2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,samplePt1,samplePt2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(samplePt1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,samplePt2,samplePt1)

    plt.figure(figsize=(15, 5))
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)

###### Compute Fundamental Matrix using OpenCV RANSAC
FRansac, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
print('Number of RANSAC inliers : ' + str(mask.sum()))
print(f"FRansac : ")
print(FRansac)

# select inlier points
inlierpts1 = pts1[mask.ravel()==1]
inlierpts2 = pts2[mask.ravel()==1]

def SampsonDist(pt1, pt2, F):
    # Calculate the Simpson's distance of a pair of points (pt1, pt2) and a
    # Fundamental matrix F
    Mpt1 = np.array([[pt1[0]], [pt1[1]], [1]])
    Mpt2 = np.array([[pt2[0]], [pt2[1]], [1]])
    num = np.dot(np.transpose(Mpt2), np.dot(F, Mpt1))
    den1 = np.dot(np.transpose(F), Mpt2)
    den2 = np.dot(F, Mpt1)
    SampDist = num[0,0]**2 / (den1[0,0]**2 + den1[1,0]**2 + den2[0,0]**2 + den2[1,0]**2)
    return SampDist

mse = 0
for inlierpt1, inlierpt2 in zip(inlierpts1, inlierpts2):
    mse += SampsonDist(inlierpt1, inlierpt2, FRansac)


rmse = math.sqrt(mse / mask.sum())
print(f"RMSE : {rmse}")

# plot epipolar lines
drawFundamental(img1,img2,inlierpts1,inlierpts2,FRansac)

###### Compute Fundamental Matrix using hand-made RANSAC

### TODO

def NSamples(sampleSize, propOutliers):
    # Calculate the number of samples required to ensure, with probability 0.99,
    # that at least one sample has no outliers for a given size of sample and
    # proportion of outliers in a RANSAC algorithm
    N_p0_99 = [[ 2, 3,  5,  6,  7,  11,   17],
               [ 3, 4,  7,  9, 11,  19,   35],
               [ 3, 5,  9, 13, 17,  34,   72],
               [ 4, 6, 12, 17, 26,  57,  146],
               [ 4, 7, 16, 24, 37,  97,  293],
               [ 4, 8, 20, 33, 54, 163,  588],
               [ 5, 9, 26, 44, 78, 272, 1177]]
    if propOutliers < 0.05:
        j = 0
    elif propOutliers < 0.1:
        j = 1
    elif propOutliers < 0.2:
        j = 2
    elif propOutliers < 0.25:
        j = 3
    elif propOutliers < 0.3:
        j = 4
    elif propOutliers < 0.4:
        j = 5
    else:
        j = 6
    return N_p0_99[sampleSize-2][j]

# RANSAC paramenters
N = 588 # Initial number of samples for p = 0.99 and proportion of outliers e = 0.5
maxDistThreshold = 2.5 # maximum distance of the solution from inliers

inliersPts1MaxN = []
inliersPts2MaxN = []
inliersPts1Max = []
inliersPts2Max = []
NinMaxN = 0
mseMin = 0
FRANSAC = np.array([[0,0,0],[0,0,0],[0,0,0]])

i = 0
propOutliers = 0.5
print("RANSAC algorithm begins...")
print("...")
while N > i:
    # Compute Fundamental Matrix F with 7-points

    indexes = np.random.randint(0, pts1.shape[0], size=(7))

    samplePt1 = pts1[indexes,:]
    samplePt2 = pts2[indexes,:]

    F7point, _ = cv2.findFundamentalMat(samplePt1, samplePt2, cv2.FM_7POINT)

    NInliers = list(range(len(F7point)//3))
    NinMax = 0

    # Choose Fundamental matrix F with more inliers in the case of having 3 solutions
    for j in range(0,len(F7point)//3):
        inliersPts1 = []
        inliersPts2 = []
        for pt1, pt2 in zip(pts1, pts2):
            dist = SampsonDist(pt1, pt2, F7point[3*j:3*(j+1),:])
#            print(f"dist : {dist}")
            if dist < maxDistThreshold:
                inliersPts1.append(pt1)
                inliersPts2.append(pt2)
        NInliers[j] = len(inliersPts1)
#        print(f"NInliers[{j+1}] : {NInliers[j]}")
        if NInliers[j] > NinMax:
            inliersPts1Max = inliersPts1
            inliersPts2Max = inliersPts2
            NinMax = NInliers[j]
            F_id = j

    # Update Fundamental matrix F, maximum number of inliers and inliers
    if NinMax > NinMaxN:
        NinMaxN = NinMax
        inliersPts1MaxN = inliersPts1Max
        inliersPts2MaxN = inliersPts2Max
        FRANSAC = F7point[3*F_id:3*(F_id+1),:]
#        print(f"FRansac 7-Point : ")
#        print(FRansac)
        print(f"N = {N}")
        print(f"i = {i}")
        print(f"e = {propOutliers}")
        print("...")

    # Adapt the number of samples N
    propOutliers = 1 - (NinMaxN)/(pts1.shape[0])
    N = NSamples(7, propOutliers)
    i += 1
print("RANSAC algorithm finishs...")

inliersPts1MaxN = np.array(inliersPts1MaxN)
inliersPts2MaxN = np.array(inliersPts2MaxN)

print(f"FRansac 7-Point optimized with RANSAC : ")
print(FRANSAC)

mse = 0
for inlierpt1, inlierpt2 in zip(inliersPts1MaxN, inliersPts2MaxN):
    mse += SampsonDist(inlierpt1, inlierpt2, FRANSAC)
rmse = math.sqrt(mse / mask.sum())
print(f"RMSE : {rmse}")

drawFundamental(img1, img2, inliersPts1MaxN, inliersPts2MaxN, FRANSAC)

FRANSAC_LM, mask = cv2.findFundamentalMat(inliersPts1MaxN, inliersPts2MaxN, cv2.FM_LMEDS)

print(f"FRansac Levenberg-Marquardt : ")
print(FRANSAC_LM)
print(f"Number of handmade RANSAC inliers : {NinMaxN}")


mse = 0
for inlierpt1, inlierpt2 in zip(inliersPts1MaxN, inliersPts2MaxN):
    mse += SampsonDist(inlierpt1, inlierpt2, FRANSAC_LM)
rmse = math.sqrt(mse / mask.sum())
print(f"RMSE : {rmse}")

drawFundamental(img1, img2, inliersPts1MaxN, inliersPts2MaxN, FRANSAC_LM)


#drawFundamental(img1,img2,inliersPt1,inliersPt2,bestF)

plt.show()
