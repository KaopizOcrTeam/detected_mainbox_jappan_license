import cv2
import imutils
import numpy as np
from lib import ultils
from math import pi
from math import atan2
def FindFourPointUsingHoughLines(edged):
    ## four points from borders of image
    LeftTop = [0, 0];
    RightTop = [edged.shape[1], 0];
    RightBot = [edged.shape[1], edged.shape[0]];
    LeftBot = [0,edged.shape[0]];
    FourBorder=[LeftTop,RightTop,RightBot,LeftBot]; ##1.tl 2.tr 3.br 4.bl

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    dilated = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel)

    ##find hough lines from edge image
    nlinesP = cv2.HoughLinesP(dilated, 1, np.pi/180, 80, minLineLength=20, maxLineGap=1)
    #nlinesP=cv2.HoughLinesP(dilated, 1, np.pi / 180, 80, 20, 1) ## runs the actual detection

    cdst = cv2.cvtColor(edged,cv2.COLOR_GRAY2BGR)
    #print(nlinesP)
    for l in nlinesP:
        cv2.line(cdst,(l[0][0],l[0][1]),(l[0][2],l[0][3]),(0,0,255),2)
    cv2.imshow('cdst',cdst)
    cv2.waitKey(0)



    if (nlinesP is None or len(nlinesP) == 1): ## no lines or one line
    
        ## can not find documents, get border of image
        border = 10;
        extLeftTop = extRightTop = extLeftBot = extRightBot =[None]*2

        extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border
        extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border
        extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border
        extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border
        return False,extLeftTop,extRightTop,extRightBot,extLeftBot
    else:
        #### have at least 2 lines
        ## find intesection points between lines
        ##check case many lines

        IntersectPoints = []
        if (len(nlinesP) > 30):
        
            print('small hough line group') 

            SmallnlinesP=ultils.getsmallsetLines(edged, nlinesP) ## remove unwanted lines

            # print small set line
            temp_img = cv2.cvtColor(edged,cv2.COLOR_GRAY2BGR)

            for l in SmallnlinesP:
                cv2.line(temp_img,(l[0][0],l[0][1]),(l[0][2],l[0][3]),(0,0,255),2)
            cv2.imshow('nncdst',temp_img)
            cv2.waitKey(0)

            if (len(SmallnlinesP)==0):
            
                print('no small line')
                ## can not find documents, get border of image
                border = 10;
                extLeftTop = extRightTop = extLeftBot = extRightBot =[None]*2

                extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border
                extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border
                extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border
                extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border
                return False,extLeftTop,extRightTop,extRightBot,extLeftBot
            
            else:
                IntersectPoints = ultils.FindPointIntersection(edged, SmallnlinesP)
        else:
            IntersectPoints = ultils.FindPointIntersection(edged, nlinesP)


        if len(IntersectPoints)<4:#less than 4 Point
            print("less than 4 Point")
            ## first set points to border points of image
            border = 10;
            extLeftTop = extRightTop = extLeftBot = extRightBot =[None]*2

            extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border
            extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border
            extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border
            extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border
            ##update new corner points 
            t1=[];
            indexP=0; ##0-tl  1-tr  3-br  3-bl
            id_closet=0;
            mindist = 0;
            maxdist_0 = maxdist_1 = maxdist_2 = maxdist_3 = 99999999

            for i in range(len(IntersectPoints)):
            
                ##update corner points of documents
                t1,indexP,mindist= ultils.Distance_ReferencePoint(IntersectPoints[i], FourBorder);

                if (indexP == 0): ##top left point

                    if (mindist <= maxdist_0):
                    
                        maxdist_0 = mindist;
                        extLeftTop = IntersectPoints[i];
                         
                elif (indexP == 1): ## top right point
                        
                    if (mindist <= maxdist_1):
                        
                        maxdist_1 = mindist;
                        extRightTop = IntersectPoints[i];     
                
                elif (indexP == 2): ## bottom right
                
                    if (mindist <= maxdist_2):
                    
                        maxdist_2 = mindist;
                        extRightBot = IntersectPoints[i];
                     
                else: ## bottom left
                    if (mindist <= maxdist_3):
                    
                        maxdist_3 = mindist;
                        extLeftBot = IntersectPoints[i];
                    
            return False,extLeftTop,extRightTop,extRightBot,extLeftBot
        else:
            print("greater than 4 Point")
            ## >= 4 points , find possible quadrilateral and get largest area
            
            Qual = ultils.FindQual(edged, IntersectPoints)
            if (Qual):
                max_area = id_x = 0
                for i,x in enumerate(Qual):
                    if(max_area<cv2.contourArea(x)):
                        id_x = i
                        max_area = cv2.contourArea(x)
                #sorted(Qual,key = lambda x: cv2.contourArea(x),reverse = True)

                SelectQual = Qual[id_x]; ##/ get max area quad  
                extLeftTop = SelectQual[0]
                extRightTop = SelectQual[1]
                extRightBot = SelectQual[2]
                extLeftBot = SelectQual[3]
                return True,extLeftTop,extRightTop,extRightBot,extLeftBot
            
            else:         
                ## get border of image
                border = 10;
                extLeftTop = extRightTop = extLeftBot = extRightBot =[None]*2

                extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border;
                extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border;
                extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border;
                extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border;
                return False,extLeftTop,extRightTop,extRightBot,extLeftBot

def FindFourCornerPoints(img):

    img_resize = img.copy()
    img_4point = img.copy()
    edged = ultils.PreProces_FindEdge(img)
    
    cv2.imshow("edged", edged);
    cv2.waitKey(0);

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    dilated = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel)

    _,contours,_h = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0] if imutils.is_cv2() else contours[1]
    
    approx = []
    for c in contours:
        peri = cv2.arcLength(c,True);
        approx.append(cv2.approxPolyDP(c,0.02* peri,True))
    # sai cho nay
    hull = []
    Group_FourCornerDoc = []
    hull_4vertice = 0
    appr_size = len(approx)

    #convex hull
    for i in range(0,appr_size):
        Area_approx = cv2.contourArea(approx[i])
        Area_image = img.shape[1] * img.shape[0]
        if (Area_approx > Area_image * 0.08):
            CHtemp = cv2.convexHull(approx[i])
            Area_hull = np.abs(cv2.contourArea(CHtemp))
            if len(CHtemp)<=8 and len(CHtemp)>=4\
            and Area_hull > Area_image * 0.15 \
            and Area_hull / Area_image < 0.9:
                #if convex hull has exactly 4 edges ... check angles and length
                if len(CHtemp) == 4:
                    #sort by order 0.top left 1. top rigth 2. bottom right 3. bottom left
                    hull_4edges  = ultils.orderPoints(CHtemp.tolist())

                    hull_4edges_angles = ultils.FindAngleFromRect(hull_4edges)
                    hull_4edges_length = ultils.FindLengthFromRect(hull_4edges)
                    
                    h_angle_min = 70;
                    h_angle_max = 110;
                    h_minedge = min(hull_4edges_length[0], hull_4edges_length[1],hull_4edges_length[2], hull_4edges_length[3])
                    h_maxedge = max(hull_4edges_length[0],hull_4edges_length[1], hull_4edges_length[2], hull_4edges_length[3])
                    if (h_minedge / h_maxedge > 0.45  and h_minedge / h_maxedge < 0.8     #0.45 0.8
                        and hull_4edges_angles[0] > h_angle_min and hull_4edges_angles[0] < h_angle_max 
                        and hull_4edges_angles[1] > h_angle_min and hull_4edges_angles[1] < h_angle_max 
                        and hull_4edges_angles[2] > h_angle_min and hull_4edges_angles[2] < h_angle_max 
                        and hull_4edges_angles[3] > h_angle_min and hull_4edges_angles[3] < h_angle_max):
                        
                        # set flag is 1
                        hull_4vertice = 1;
                        hull.append(hull_4edges) #dont find rotated rectangle for that hull
                        continue
                
                #### find rotated rectangle of convex hull (number of edgeds > 4)
                TempRect = cv2.minAreaRect(approx[i]);
                rectsize = len(TempRect)
                rect_points = cv2.boxPoints(TempRect); # get 4 points from rotated rect
                
                xrank = np.argsort(rect_points[:, 0])

                left = rect_points[xrank[:2], :]
                yrank = np.argsort(left[:, 1])
                left = left[yrank, :]

                right = rect_points[xrank[2:], :]
                yrank = np.argsort(right[:, 1])
                right = right[yrank, :]

                #            top-left,       top-right,       bottom-right,    bottom-left
                box_coords = tuple(left[0]), tuple(right[0]), tuple(right[1]), tuple(left[1])
                box_height = ultils._distance(left[0],left[1])
                box_width = ultils._distance(left[0],right[0])

                ratio_m = min(box_height, box_width) / max(box_height, box_width)
                
                if (ratio_m > 0.45 and ratio_m < 0.8
                    and left[0][0] > 0 and left[0][1] < img.shape[0] 
                    and right[0][0]>0 and right[0][1] > 0 
                    and right[1][0] < img.shape[1] and right[1][1]>0 
                    and left[1][0] < img.shape[1] and left[1][1] < img.shape[0]):

                    RRectPoint=[rect_points[0],rect_points[1],rect_points[2],rect_points[3]]
                    orderRRect = ultils.orderPoints(RRectPoint)
                    Group_FourCornerDoc.append(orderRRect)
                ###delete if coordinate value of points is negative and ratio between heigh and width
    #sort hull with 4 edges on their areas, the first is the largest one
    
    hull = np.array(hull).astype(np.int32)
    hull = sorted(hull, key = lambda x: np.abs(cv2.contourArea(x)),reverse=True)
    #sort rotated rectangle based on their area, the first is the largest one
    Group_FourCornerDoc = np.array(Group_FourCornerDoc).astype(np.int32)

    Group_FourCornerDoc = sorted(Group_FourCornerDoc,key = lambda x: np.abs(cv2.contourArea(x)),reverse=True)

    convexHull_mask = np.zeros_like(img)

    for i in range(0,len(hull)):
        cv2.drawContours(convexHull_mask,hull,i, (255,255,255), 3)
        print("area of convex hull :" ,np.abs(cv2.contourArea(hull[i])))

    cv2.imshow("convexHull_mask", convexHull_mask);
    cv2.waitKey(0);


    convexHull_mask2 = np.zeros_like(img)

    for i in range(0,len(Group_FourCornerDoc)):
        cv2.drawContours(convexHull_mask2,Group_FourCornerDoc,i, (255,255,255), 3)

    cv2.imshow("rotated retangle", convexHull_mask2);
    cv2.waitKey(0);

    extLeftTop, extRightTop, extLeftBot, extRightBot = None,None,None,None

    if (len(Group_FourCornerDoc)==0 and len(hull)==0):
        print("case 1: no enclosed contour")
        #apply the Algorithm 1
        #(Algorithm 1) Find possible four corners of document based on hough lines using imgage edges.
        _,extLeftTop, extRightTop, extLeftBot, extRightBot = FindFourPointUsingHoughLines(edged)
    #convex hull with four edged
    elif hull_4vertice == 1:
        print("case 2: hull 4 edged")
        # compare ratio with  convex hull using houglines method (algorithm 1)
        check_HL, extLeftTopHL, extRightTopHL, extRightBotHL, extLeftBotHL=FindFourPointUsingHoughLines(edged)
        FourextPointfromHL=np.array([extLeftTopHL, extRightTopHL, extRightBotHL, extLeftBotHL]).astype(np.int32) #1.tl 2.tr 3.br 4.bl
        # get the smallest one from hull 4 edged
        extLeftTop,extRightTop,extRightBot,extLeftBot = hull[-1][0],hull[-1][1],hull[-1][2],hull[-1][3]
        # update four corners if possible
        if check_HL is True:
            FourextPointfromHULL = np.array([extLeftTop,extRightTop,extRightBot,extLeftBot]).astype(np.int32)
            # compare ratio between two areas
            ratio2Area = np.abs(cv2.contourArea(FourextPointfromHULL)) / np.abs(cv2.contourArea(FourextPointfromHL))
            print("abs(contourArea(Mat(FourextPointfromHL))) / (edged.cols*edged.rows)" 
                ,np.abs(cv2.contourArea(FourextPointfromHL)) / (edged.shape[0]*edged.shape[1]))
            if (ratio2Area < 0.5 or ratio2Area >1) and np.abs(cv2.contourArea(FourextPointfromHL)) / (edged.shape[0]*edged.shape[1])<0.75:
                #get four corners from hough line method (algorithm 1)
                extLeftTop,extRightTop,extRightBot,extLeftBot = extLeftTopHL,extRightTopHL,extRightBotHL,extLeftBotHL


    elif len(Group_FourCornerDoc) == 1:
        print("case 3: only one rect")
        #compare ratio with convex hull using houglines method (algoritm 1)
        #Point extLeftTopHL, extRightTopHL, extLeftBotHL, extRightBotHL;
        check_HL, extLeftTopHL, extRightTopHL, extRightBotHL, extLeftBotHL=FindFourPointUsingHoughLines(edged)
        #1.tl 2.tr 3.br 4.bl
        FourextPointfromHL=np.array([extLeftTopHL, extRightTopHL, extRightBotHL, extLeftBotHL]).astype(np.int32) 

        #get the largest one from set of rotated rectangles
        extLeftTop,extRightTop,extRightBot,extLeftBot = Group_FourCornerDoc[0][0],Group_FourCornerDoc[0][1],Group_FourCornerDoc[0][2],Group_FourCornerDoc[0][3]
        if check_HL is True:
            FourextPointfromHULL = np.array([extLeftTop,extRightTop,extRightBot,extLeftBot]).astype(np.int32)
            # compare ratio between two areas;
            ratio2Area = np.abs(cv2.contourArea(FourextPointfromHULL)) / np.abs(cv2.contourArea(FourextPointfromHL))
            if (ratio2Area < 0.5 or ratio2Area >1) and np.abs(cv2.contourArea(FourextPointfromHL)) / (edged.shape[0]*edged.shape[1])<0.75:
                #get four corners from hough line method (algorithm 1)
                extLeftTop,extRightTop,extRightBot,extLeftBot = extLeftTopHL,extRightTopHL,extRightBotHL,extLeftBotHL


    else:
        print("case 4: multi rects")
        
        if(len(Group_FourCornerDoc)== 2):

            # hỏi lại chổ n
            extLeftTop, extRightTop, extRightBot, extLeftBot = Group_FourCornerDoc[1][0],Group_FourCornerDoc[1][1],Group_FourCornerDoc[1][2],Group_FourCornerDoc[1][3]
            #check inside the contour
            btl = cv2.pointPolygonTest(Group_FourCornerDoc[0], (extLeftTop[0],extLeftTop[1]), False);
            brt = cv2.pointPolygonTest(Group_FourCornerDoc[0], (extRightTop[0],extRightTop[1]), False);
            brb = cv2.pointPolygonTest(Group_FourCornerDoc[0], (extRightBot[0],extRightBot[1]), False);
            blb = cv2.pointPolygonTest(Group_FourCornerDoc[0], (extLeftBot[0],extLeftBot[1]), False);

            if (btl > -1 and brt > -1 and brb > -1 and blb > -1):# // inside
                #get corners from the largest one;
                extLeftTop = Group_FourCornerDoc[0][0];
                extRightTop = Group_FourCornerDoc[0][1];
                extRightBot = Group_FourCornerDoc[0][2];
                extLeftBot = Group_FourCornerDoc[0][3];


        else:
            # multiple RECTs : find intersection points and largest quadrilateral 
            LeftTop,RightTop,LeftBot,RightBot = [0,0],[edged.shape[1], 0],[0,edged.shape[0]],[edged.shape[1],edged.shape[0]]
            for i in range(len(Group_FourCornerDoc)):
            
                ##get 4 lines for each RECT
                extl = Group_FourCornerDoc[i][0];
                exrt = Group_FourCornerDoc[i][1];
                exrb = Group_FourCornerDoc[i][2];
                exlb = Group_FourCornerDoc[i][3];
                l1=[None]*4
                l2=[None]*4
                l3=[None]*4
                l4=[None]*4
                l1[0] = extl[0]; l1[1] = extl[1]; l1[2] = exrt[0]; l1[3] = exrt[1];
                l2[0] = exrt[0]; l2[1] = exrt[1]; l2[2] = exrb[0]; l2[3] = exrb[1];
                l3[0] = exrb[0]; l3[1] = exrb[1]; l3[2] = exlb[0]; l3[3] = exlb[1];
                l4[0] = exlb[0]; l4[1] = exlb[1]; l4[2] = extl[0]; l4[3] = extl[1];
                nlinesP.append(l1);
                nlinesP.append(l2);
                nlinesP.append(l3);
                nlinesP.append(l4);
            
            if (not nlinesP): ## no line
            
                border = 10;
                extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border;
                extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border;
                extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border;
                extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border;
            
            else:
            
                ## find intesection points
                
                IntersectPoints=FindPointIntersection(edged, nlinesP);
                Qual=FindQual(edged, IntersectPoints);
                if (Qual):
                
                    ##cout << 'number of quadrilaterals' << Qual.size() << endl;
                    sort(Qual[0], Qual[0], compareContourAreas);
                    SelectQual = Qual[0];
                    extLeftTop = SelectQual[0];
                    extRightTop = SelectQual[1];
                    extRightBot = SelectQual[2];
                    extLeftBot = SelectQual[3];
                
                else:
                
                    
                    extLeftTop[0] = LeftTop[0] + border; extLeftTop[1] = LeftTop[1] + border;
                    extRightTop[0] = RightTop[0] - border;extRightTop[1] = RightTop[1] + border;
                    extLeftBot[0] = LeftBot[0] + border;extLeftBot[1] = LeftBot[1] - border;
                    extRightBot[0] = RightBot[0] - border;extRightBot[1] = RightBot[1] - border;


    cv2.circle(img_4point, tuple(extLeftTop), 5, (0, 255, 0), -1)
    cv2.circle(img_4point, tuple(extLeftBot), 5, (0, 150, 255), -1)
    cv2.circle(img_4point, tuple(extRightBot), 5, (0, 0, 255), -1)
    cv2.circle(img_4point, tuple(extRightTop), 5, (255, 50, 255), -1)

    cv2.imshow('Four_Points', img_4point);
    cv2.waitKey(0);
    cv2.imshow('wrap',ultils.four_point_transform(img,[extLeftTop,extLeftBot,extRightBot,extRightTop]))
    cv2.waitKey(0);
    FourPoints=[]
    FourPoints.append(extLeftTop);
    FourPoints.append(extRightTop);
    FourPoints.append(extLeftBot);
    FourPoints.append(extRightBot);
    #return []

def main():
    #print(ultils.orderPoints([[[5,5]],[[0,0]],[[5,0]],[[0,5]]]))
    img = cv2.imread('pic10.jpg',1)
    img = ultils.resizeToHeight(img,500)

    ratio = img.shape[1] / 500.0
    areaImage = img.shape[0]*img.shape[1]

    temp = FindFourCornerPoints(img)

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
