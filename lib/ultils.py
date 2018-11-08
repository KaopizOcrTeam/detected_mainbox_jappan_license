import cv2 
import numpy as np

from itertools import combinations
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from matplotlib import pyplot as plt
from math import pi
from math import atan2
from math import ceil

##cross product len3array
def cross(a , b):
    result=[]
    result.append(a[1] * b[2] - a[2] * b[1])
    result.append(a[2] * b[0] - a[0] * b[2])
    result.append(a[0] * b[1] - a[1] * b[0])
    return result

def resizeToHeight(img,h):
    height, width, depth = img.shape
    return cv2.resize(img,(int(img.shape[1]*h/height),int(h)),interpolation = cv2.INTER_AREA)

def PreProces_FindEdge(origin_img):
    img = cv2.cvtColor(origin_img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lower_threshold = max(0, int(ret*0.1))
    upper_threshold = min(255, int(ret))

    return cv2.Canny(img,ret*0.1,ret)


def _distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


#sort point follows order 1. top left  2. top right 3. bot right 4. bot left

def orderPoints(listPoint):

    #convert listPoint to normal form
    k = [it[0] if isinstance(it[0],list) else it for it in listPoint]
    #k = listPoint
    k = sorted(k,key=lambda x: x[0])
    lm = k[0:2];rm = k[2:4]
    lm = sorted(lm ,key=lambda x:x[1])
    tl = lm[0];bl = lm[1]
    tmp = []
    for i in range(0,2):
        tmp.append([tl, rm[i]])
    tmp = sorted(tmp,key=lambda x:_distance(x[0],x[1]))
    tr = tmp[0][1];br = tmp[1][1]
    return [tl,tr,br,bl]

#find angle between two lines P1P2 and P2P3 

def FindAngleFromTwoLines(p0,p1,p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    result = np.abs(np.degrees(angle))
    return result if result<=180.0 else 360.0-result
#Find angles between edges of quadrilateral
def FindAngleFromRect(listPoint):
    tl,tr,br,bl = listPoint
    return [FindAngleFromTwoLines(bl, tl, tr),
            FindAngleFromTwoLines(tl, tr, br),
            FindAngleFromTwoLines(tr, br, bl),
            FindAngleFromTwoLines(br, bl, tl)
           ]


#Find possible quadrilateral 
def FindQual(edged,IntersectPoints):
    n=len(IntersectPoints) ## number of intersection points
    s=range(n)

    z=list(combinations(s,4))
    Qual = []
    for i in range(len(z)):
    
        ## get one combination
        KK = z[i]
        inpts=[]
        inpts.append(IntersectPoints[KK[0]])
        inpts.append(IntersectPoints[KK[1]])
        inpts.append(IntersectPoints[KK[2]])
        inpts.append(IntersectPoints[KK[3]])


        ordered=orderPoints(inpts) ##0 tl ; 1 tr; 2 br; 3 bf

        angles = FindAngleFromRect(ordered)

        ## find distance between vertices of quad
        distance_01 = _distance(ordered[0], ordered[1])
        distance_12 = _distance(ordered[2], ordered[1])
        distance_23 = _distance(ordered[2], ordered[3])
        distance_30 = _distance(ordered[0], ordered[3])
        minedge = min(distance_01,distance_12,distance_23, distance_30)
        maxedge = max(distance_01,distance_12,distance_23, distance_30)

        cnt  = [np.array(ordered, dtype=np.int32)]
        areaQual = np.abs(cv2.contourArea(cnt[0]))## area of quad
        angle_min = 75;
        angle_max = 110;

        ##Mat cdst;
        ##cvtColor(edged, cdst, COLOR_GRAY2BGR);    
        ####/ small (minedge / maxedge) 0.45 for rotated input ,,  0.55 for front input
        if (minedge / maxedge > 0.45 and minedge / maxedge < 0.8 
            and areaQual > len(edged[0])*len(edged) * 0.08 
            and areaQual / (len(edged[0])*len(edged)) < 0.9 
            and angles[0] > angle_min  and angles[0] < angle_max 
            and angles[1] > angle_min and angles[1] < angle_max 
            and angles[2] > angle_min and angles[2] < angle_max 
            and angles[3] > angle_min and angles[3] < angle_max):


            Qual.append(cnt[0])

    return Qual
# Find length of edges in quadrilateral 
def FindLengthFromRect(listPoint):
    tl,tr,br,bl = listPoint
    return [_distance(tl, tr),_distance(tr, br),_distance(br, bl),_distance(bl, tl)]

def getsmallsetLines(edged, nlinesP):

    sortXnlinesP=[]## horizontal lines
    sortYnlinesP=[]## vertical lines

    for nlinesPr1 in nlinesP:
    
        l = nlinesPr1

        angle1 = atan2(l[0][1] - l[0][3], l[0][0] - l[0][2])
        result = angle1 * 180.0 / pi
        
        if (result < 0):

            result = abs(result)
        
        if (result > 180.0):
        
            result = 360.0 - result
        
        if (abs(l[0][0] - l[0][2]) > abs(l[0][1] - l[0][3])): ## horizontal lines    
            
            if (result > 160 or result <30):
                sortXnlinesP.append(nlinesPr1)

        else: ##/ vertical  lines
        
            if (result > 75 and result < 115):
            
                sortYnlinesP.append(nlinesPr1);
            
    sorted(sortXnlinesP,key = lambda x:x[0][1]) #sort compare Y Line
    sorted(sortYnlinesP,key = lambda x:x[0][0]) #sort compare X Line

    minsizeXY = min(len(sortXnlinesP), len(sortYnlinesP));
    if (minsizeXY < 4):
        tempsize = minsizeXY;
    else:
        tempsize = minsizeXY / 3;
    limitsize = min(7, tempsize);
    SmallnlinesP=[]
    for r2 in range(0,int(limitsize)): ## get almost 7 lines from each border of image , remove most of lines near center
    
        SmallnlinesP.append(sortXnlinesP[r2]);
        SmallnlinesP.append(sortXnlinesP[len(sortXnlinesP) - r2 - 1]);
        SmallnlinesP.append(sortYnlinesP[r2]);
        SmallnlinesP.append(sortYnlinesP[len(sortYnlinesP) - r2 - 1]);
    
    return SmallnlinesP

#Find distance between a reference point  and other points in a group
def Distance_ReferencePoint(refer,GroupPoint):
    k=0
    mindist=_distance(refer, GroupPoint[k])
    for i in range(len(GroupPoint)):
    
        dist = _distance(refer, GroupPoint[i])
        if dist<mindist:
            mindist = dist
            k = i

    return GroupPoint[k],k,mindist

#.. get intersectionPoint and angle between line_a and line_b
def get_intersection(line_a,line_b):

    pa = [line_a[0][0], line_a[0][1], 1]
    pb = [line_a[0][2], line_a[0][3], 1]
    la = cross(pa, pb)
    pa[0] = line_b[0][0]; pa[1] = line_b[0][1]; pa[2] = 1
    pb[0] = line_b[0][2]; pb[1] = line_b[0][3]; pb[2] = 1
    lb = cross(pa, pb)
    inter = cross(la, lb)
    if (inter[2] == 0):
        return False, None, None ## two lines are parallel
    else:
        intersection=[None]*2
        intersection[0] = inter[0] / inter[2]
        intersection[1] = inter[1] / inter[2]
        angle1 = atan2(line_a[0][1] - line_a[0][3], line_a[0][0] - line_a[0][2])
        angle2 = atan2(line_b[0][1] - line_b[0][3], line_b[0][0] - line_b[0][2])
        result = (angle2 - angle1) * 180.0 / pi;
        if (result < 0):
    
            result = abs(result);

        if (result > 180.0):
        
            result = 360.0 - result;
        
        return True,intersection,result

def FindPointIntersection(edged, linesP):

    ## loop over lines
    
    lines_size = len(linesP);
    IntersectPoints=[]
    for r1 in range(lines_size-1):
    
        l = linesP[r1];
        ## loop over remaining lines
        for r2 in range(r1+1,lines_size):
        
            ll = linesP[r2];
            ## check for intersection
            is_intersect,Intersect,angle=get_intersection(l, ll)

            if (is_intersect):

                if (Intersect[0] >= 0 and Intersect[0] <= len(edged[0]) 
                    and Intersect[1] >= 0 and Intersect[1] <= len(edged) 
                    and angle >= 85.0 and angle <= 105.0):#105
                    
                    IntersectPoints.append(Intersect)

    if (IntersectPoints):
        
        th_distance = 25
        clustering=DBSCAN(eps=th_distance,min_samples=1).fit(IntersectPoints)
        #n_labels = partition(IntersectPoints, Cluslabels, [th2](const Pointand lhs, const Pointand rhs) 
            #return ((lhs[0] - rhs[0])*(lhs[0] - rhs[0]) + (lhs[1] - rhs[1])*(lhs[1] - rhs[1])) < th2;});
        ## You can save all points in the same class in a vector (one for each class)
        n_labels=max(clustering.labels_+1)
        ClusterPoint=[None]*n_labels

        for i in range(len(IntersectPoints)):
            if clustering.labels_[i]>-1:
                if ClusterPoint[clustering.labels_[i]] is None:
                    ClusterPoint[clustering.labels_[i]]=[IntersectPoints[i]]
                else:

                    ClusterPoint[clustering.labels_[i]].append(IntersectPoints[i])
                    
        ## Find mean of each cluster 
        NewIntersectPoints=[]
        for j in range(n_labels):

            #sumx = sumy = 0
            sumx=sumy=0
            for c in ClusterPoint[j]:
                sumx+=c[0]
                sumy+=c[1]
            
            meanvec=[ceil(sumx / len(ClusterPoint[j])),ceil(sumy / len(ClusterPoint[j]))]

            NewIntersectPoints.append(meanvec);

        print("new number of intesection Points:",len(NewIntersectPoints))
        return NewIntersectPoints
    return []
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = orderPoints(pts)
    tl, tr, br, bl = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.float32([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]])

    rect = np.float32(rect)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped