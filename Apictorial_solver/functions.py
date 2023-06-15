import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rng
from scipy.optimize import linear_sum_assignment

def Load_images(num, invert=False, binary=False):
    imagearray = []
    imagearray_gray = []
    for i in range(num):
        path = "./Moomie/Cropped_ImageCroppedImage{}.jpg".format(i)
        img = cv2.imread(path)
        if invert:
            img = cv2.flip(img,1)
        img_blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        if binary:
            mask = BinaryMask(img_gray)
            img = cv2.bitwise_and(img, img, mask=mask)
        imagearray.append(img)
        imagearray_gray.append(img_gray)

    return imagearray, imagearray_gray

def CornerDetection(imagename, drawCircle=False):
    input = imagename.copy()
    #blur = cv2.GaussianBlur(input, (7, 7), cv2.BORDER_DEFAULT)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    h, w = input.shape
    orig_y = h / 2
    orig_x = w / 2

    corners = cv2.goodFeaturesToTrack(input, 4, 0.01, 300)
    corners = np.int0(corners)
    cnr = []
    # res = []
    for i in corners:
        x, y = i.ravel()
        if drawCircle == True:
            cv2.circle(input, (x, y), 7, (255, 0, 0), -1)
        cnr.append([x, y])

    n_cnr = [[coord[0] - orig_x, orig_y - coord[1]] for coord in cnr]

    quad1, quad2, quad3, quad4 = [], [], [], []
    for coord in n_cnr:
        if coord[0] >= 0 and coord[1] >= 0:
            quad1.append(coord)
        elif coord[0] < 0 and coord[1] >= 0:
            quad2.append(coord)
        elif coord[0] < 0 and coord[1] < 0:
            quad3.append(coord)
        else:
            quad4.append(coord)

    quad1 = sorted(quad1, key=lambda c: c[0])
    quad2 = sorted(quad2, key=lambda c: c[0], reverse=True)
    quad3 = sorted(quad3, key=lambda c: c[0], reverse=True)
    quad4 = sorted(quad4, key=lambda c: c[0])

    res = quad2 + quad1 + quad3 + quad4

    res = [[int(coord[0] + orig_x), int(- coord[1] + orig_y)] for coord in res]

    return input, res

def rotationAboutFixedAxis(img,pointOfRotation,secondPoint):
    """
    Calculates the angle of rotation and rotates the image so that the two points lie on the same y-axis.
    
    :param img: Image
    :param pointOfRotation: Specify point of rotation in tuple(int(x),int(y)) coordinates.
    :param secondPoint: Point to be rotated (x,y)
    
    :return: Rotated image
    
    """
    angle = math.atan((secondPoint[1]-pointOfRotation[1])/(secondPoint[0]-pointOfRotation[0]))*180/math.pi
    M = cv2.getRotationMatrix2D(pointOfRotation,angle,1)
    dst = cv2.warpAffine(img,M,(0,0))
    
    return dst

def edgeFeatureExtraction(img):
    """
    Function that finds the edge contours in an image and
    returns a cropped image with only the dominant edge
    
    :param img: Binary image of cropped side piece
    
    :return: image containing only the longest edge. 
    
    """
    
    edge = cv2.Canny(img,threshold1=200,threshold2=700)
    contours,_ = cv2.findContours(edge,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if (len(contours)!=0):
        top_contour = [len(i) for i in contours]
        index = np.argmax(np.array(top_contour))
        for i in range(len(contours)):
            if i == index:
                pass
            else:
                cv2.boundingRect(contours[i])       
                x,y,w,h = cv2.boundingRect(contours[i])
                img[y:y+h,x:x+w] = 255

        return img
    else:
        return img

# def matchEdges(side_img,side_img2,plot=False,shifted_img=False):
#     '''
#     Function that takes two binary side images,
#     translate them to same coordinates and overlay them to check for match. 

#     param edge: Binary image of hole
#     param edge2: Binary image of head

#     return: number of white pixels in the overlay image (smaller number would suggest better match)
#     '''

#     side_img = cv2.resize(side_img,(400,200))
#     side_img2 = cv2.resize(side_img2,(400,200))
    
#     side_img2 = cv2.rotate(side_img2,cv2.ROTATE_180)
#     edge = cv2.Canny(side_img,threshold1=200,threshold2=700)
#     edge2 = cv2.Canny(side_img2,threshold1=200,threshold2=700)
        
#     # For finding first nonzero element in the first column
#     x1 = 0
#     x2 = 0
#     for i in range(len(edge[:,0])):
#         if edge[i,0]==255:
#             x1 = i
#         if edge2[i,0]==255:
#             x2 = i
#         if x1>0 and x2>0:
#             break
    
#     M = np.float32([
#         [1, 0, 0],
#         [0, 1 , x1-x2]
#     ])

#     shifted = cv2.warpAffine(side_img2,M,(side_img.shape[1],side_img2.shape[0]))
#     overlay = cv2.bitwise_xor(side_img,shifted)
#     overlay = cv2.bitwise_not(overlay)
#     if plot==True:
#         plt.imshow(overlay,cmap='gray')
#     if shifted_img:
#         return shifted,side_img
#     else:
#         return np.count_nonzero(overlay)
    
def matchEdges(side_img, side_img2, plot=False, shifted_img=False):
    '''
    Function that takes two binary side images,
    translate them to same coordinates and overlay them to check for match. 

    param edge: Binary image of hole
    param edge2: Binary image of head

    return: number of white pixels in the overlay image (smaller number would suggest better match)
    '''

    
    side_img = cv2.resize(side_img, (400, 200))
    side_img2 = cv2.resize(side_img2, (400, 200))

    side_img2 = cv2.rotate(side_img2, cv2.ROTATE_180)
    edge = cv2.Canny(side_img, threshold1=200, threshold2=700)
    edge2 = cv2.Canny(side_img2, threshold1=200, threshold2=700)

    # For finding first nonzero element in the first column
    x1 = np.argmax(edge[:, 0] == 255)
    x2 = np.argmax(edge2[:, 0] == 255)

    M = np.float32([
        [1, 0, 0],
        [0, 1, x1 - x2]
    ])

    shifted = cv2.warpAffine(side_img2, M, (side_img.shape[1], side_img2.shape[0]))
    overlay = cv2.bitwise_xor(side_img, shifted)
    overlay = cv2.bitwise_not(overlay)
    
    if overlay[0,0] != 0:
        length = np.argwhere(overlay[:,0]==0)
        overlay[0:int(length[0]),0:400] = 0 

    if plot:
        plt.imshow(overlay, cmap='gray')

    if shifted_img:
        return shifted, side_img
    else:
        return np.count_nonzero(overlay)


    
def HullContour(img,fig=False):
    '''
    :param img: Binary image
    :param fig: If True plots the contours and hull

    :return: Contour and list containing hull

    '''
    img_edge = img.copy()
    edge = cv2.Canny(img_edge,threshold1=200,threshold2=700)
    contours,_ = cv2.findContours(edge,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color=(0,0,256))
        cv2.drawContours(drawing, hull_list, i, color)

    if fig==True:
        plt.figure(figsize=(8,10))
        plt.imshow(drawing)

    return contours, hull_list


def ConvexDefect(img,defectpoints=False):
    '''
    :param img: Binary image
    :return: Color image with defect and convex
    '''

    imagename = img.copy()
    #imagename = BinaryMask(imagename)
    contours, hierarchy = cv2.findContours(imagename, 2, 1)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    defs = []
    length_from_hull = []
    far_point = []
    for i in range(defects.shape[0]):
        if defects[i][0][3] > 5000:
            defs.append(defects)
            s, e, f, d, = defects[i, 0]
            far = tuple(cnt[f][0])
            cv2.circle(imagename, far, 5, [0, 0, 255], -1)
            length_from_hull.append(defects[i][0][3]/256)
            far_point.append(far)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])

        cv2.line(imagename, start, end, [0, 255, 255], 2)
        
    if defectpoints:
        return far_point
    else:
        return length_from_hull

def BinaryMask(imgname):
    input = imgname.copy()
    thresh = 100
    maxValue = 255
    th, dst = cv2.threshold(input, thresh, maxValue, cv2.THRESH_OTSU)

    return dst

def cropp_image_sides(img, ylim_min, ylim_max, mask=False, plot=False,gray_plot=False):
    '''
    Cropps every side of the puzzle piece. The function does the cropping on x-axis based on cornerdetection.
    
    :param img: Image of a puzzle piece
    :param ylim_min: From y-value
    :param ylim_max: To y-value
    :param mask: Masks the shortest edges
    :param plot: Plots the binary images of all sides
    :param gray_plot: Plots the gray image of all sides
    
    :return: top- , left-, right-, bottom-image
    
    '''
    img_copi = img.copy()

    if len(img.shape) == 3:
        img_copi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Detect corners in the image
    _, cnr1_ = CornerDetection(img_copi)
    img_rot = rotationAboutFixedAxis(img_copi, (int(cnr1_[0][1]), int(cnr1_[0][1])), cnr1_[1])
    _, cnr1 = CornerDetection(img_rot)

    img90c = cv2.rotate(img_copi, cv2.ROTATE_90_CLOCKWISE)
    _, cnr2_ = CornerDetection(img90c)
    img_rot_90c = rotationAboutFixedAxis(img90c, (int(cnr2_[0][0]), int(cnr2_[0][1])), cnr2_[1])
    _, cnr2 = CornerDetection(img_rot_90c)
    
    img180c = cv2.rotate(img90c, cv2.ROTATE_90_CLOCKWISE)
    _, cnr3_ = CornerDetection(img180c)
    img_rot180c = rotationAboutFixedAxis(img180c, (int(cnr3_[0][0]), int(cnr3_[0][1])), cnr3_[1])
    _, cnr3 = CornerDetection(img_rot180c)
    
    img270c = cv2.rotate(img180c, cv2.ROTATE_90_CLOCKWISE)
    _, cnr4_ = CornerDetection(img270c)
    img_rot270c = rotationAboutFixedAxis(img270c, (int(cnr4_[0][0]), int(cnr4_[0][1])), cnr4_[1])
    _, cnr4 = CornerDetection(img_rot270c)

    
    if len(img.shape) == 3:
        img_rot_col = rotationAboutFixedAxis(img, (int(cnr1_[0][1]), int(cnr1_[0][1])), cnr1_[1])        
        topside, topSide_col = colorThresholder_hsv(img_rot_col[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]])

        img90c_col = cv2.rotate(img_copi, cv2.ROTATE_90_CLOCKWISE)
        img_rot_90c_col = rotationAboutFixedAxis(img90c_col, (int(cnr2_[0][0]), int(cnr2_[0][1])), cnr2_[1])
        leftSide, leftSide_col = colorThresholder_hsv(img_rot_90c_col[ylim_min:ylim_max, cnr2[0][0]:cnr2[1][0]])

        img180c_col = cv2.rotate(img90c_col, cv2.ROTATE_90_CLOCKWISE)
        img_rot180c_col = rotationAboutFixedAxis(img180c_col, (int(cnr3_[0][0]), int(cnr3_[0][1])), cnr3_[1])
        bottomSide, bottomSide_col = colorThresholder_hsv(img_rot180c_col[ylim_min:ylim_max, cnr3[0][0]:cnr3[1][0]])
        
        img270c_col = cv2.rotate(img180c_col, cv2.ROTATE_90_CLOCKWISE)
        img_rot270c_col = rotationAboutFixedAxis(img270c_col, (int(cnr4_[0][0]), int(cnr4_[0][1])), cnr4_[1])
        rightSide, rightSide_col = colorThresholder_hsv(img_rot270c_col[ylim_min:ylim_max, cnr4[0][0]:cnr4[1][0]])

    else:
        topSide = BinaryMask(img_rot[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]])
        leftSide = BinaryMask(img_rot_90c[ylim_min:ylim_max, (cnr2[0][0]):cnr2[1][0]])
        bottomSide = BinaryMask(img_rot180c[ylim_min:ylim_max, cnr3[0][0]:cnr3[1][0]])
        rightSide = BinaryMask(img_rot270c[ylim_min:ylim_max, (cnr4[0][0]):cnr4[1][0]])

    if mask == True:
        topSide = edgeFeatureExtraction(topSide)
        leftSide = edgeFeatureExtraction(leftSide)
        rightSide = edgeFeatureExtraction(rightSide)
        bottomSide = edgeFeatureExtraction(bottomSide)

    if plot == True:
        plt.figure(figsize=(8, 5))
        plt.subplot(221)
        plt.title('Topside')
        plt.imshow(topSide, cmap='gray')
        plt.subplot(222)
        plt.title('Leftside')
        plt.imshow(leftSide, cmap='gray')
        plt.subplot(223)
        plt.title('Rightside')
        plt.imshow(rightSide, cmap='gray')
        plt.subplot(224)
        plt.title('Bottomside')
        plt.imshow(bottomSide, cmap='gray')
        plt.show()
    
    if gray_plot == True:
        topSide_gray = img_rot[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]]
        leftSide_gray = img_rot_90c[ylim_min:ylim_max, cnr2[0][0]:cnr2[1][0]]
        bottomSide_gray = img_rot180c[ylim_min:ylim_max, cnr3[0][0]:cnr3[1][0]]
        rightSide_gray = img_rot270c[ylim_min:ylim_max, cnr4[0][0]:cnr4[1][0]]
        
        plt.figure(figsize=(8, 5))
        plt.subplot(221)
        plt.title('Topside')
        plt.imshow(topSide_gray, cmap='gray')
        plt.subplot(222)
        plt.title('Leftside')
        plt.imshow(leftSide_gray, cmap='gray')
        plt.subplot(223)
        plt.title('Rightside')
        plt.imshow(rightSide_gray, cmap='gray')
        plt.subplot(224)
        plt.title('Bottomside')
        plt.imshow(bottomSide_gray, cmap='gray')
        plt.show()
    
    if len(img.shape) == 3:
        return topSide, topSide_col, leftSide, leftSide_col, rightSide, rightSide_col, bottomSide, bottomSide_col
    else:
        return topSide, leftSide, rightSide, bottomSide


def SideType(img):
    '''
    For checking what type of side the img belongs to

    :param: Cropped image of one side of puzzle piece

    :return: Integer from 0 - 2 depending on which type of sidetype it belongs to. 
    '''

    defects = ConvexDefect(img)
    
    if len(defects)==0:
        return int(0)
    elif len(defects)==1:
        return int(-1)
    elif len(defects)==2:
        return int(1)


def ReadPieceSides(imagearray):
    pieceSides = []
    piecetype = np.zeros((len(imagearray),4))
    for i in range(len(imagearray)):
        pieceSides.append(i)
    for i in range(len(pieceSides)):
        try:
            t,l,r,b = cropp_image_sides(imagearray[i],0,160,mask=True)
            pieceSides[i] = t,r,l,b
            piecetype[i] = (SideType(t),SideType(l),SideType(r),SideType(b))
        except:
            print('Fail to load side image in image nr {}'.format(i))

    return pieceSides, piecetype

def ReadPieceSides2(imagearray):
    '''
    A function that reads an array of images an returns the cropped version of each side in
    jigsaw piece and also returns an array for which type of piece (head(1),hole(-1),border(0))
    the corresponding image belongs to.

    :param imagearray: an array containing a gray scale image of each puzzle piece

    :return: cropped images of sides, piecetype array 

    '''
    pieceSides = [cropp_image_sides(img, 0, 250, mask=True) for img in imagearray]
    piecetype = np.array([SideType(side) for sides in pieceSides for side in sides])
    piecetype = piecetype.reshape(len(imagearray), 4)

    return pieceSides, piecetype

def SortRelevantBorderSide(border_side_type):
    '''
    A function that sorts out the relevant sides for the borderpieces.
    Used for exlcuding the sides in the border type pieces that should not 
    be included in the border assembly

    :param border_side_type: An array containing the arrangement of the borderside
                             0 for edge, 1 for hole, -1 for head.
                             
    :return: An updated array with where the opposite side from the edge has been set to 0.
    
    '''
    
    list_of_relevant = []
    
    for i in range(len(border_side_type)):
        list_of_sides = [x for x in border_side_type[i]]
        
        if 0 in border_side_type[i]:
            indices = np.where(border_side_type[i] == 0)[0]
            
            if len(indices) == 1:
                index = indices[0]
                
                if border_side_type[i][0] == 0 or border_side_type[i][3] == 0:
                    list_of_sides[0] = 0
                    list_of_sides[3] = 0
                elif border_side_type[i][1] == 0 or border_side_type[i][2] == 0:
                    list_of_sides[1] = 0
                    list_of_sides[2] = 0
                    
            else:
                index = indices.tolist()
                
                if 0 in index and 3 in index:
                    list_of_sides[0] = 0
                    list_of_sides[3] = 0
                elif 1 in index and 2 in index:
                    list_of_sides[1] = 0
                    list_of_sides[2] = 0
        
        list_of_relevant.append(list_of_sides)

    return np.array(list_of_relevant)


def SortByType(side_type):
    '''
    Sorts the side types into two categories; interior pieces and border pieces.
    
    :param side_type: An array that contains the combination of hole/head/border for each piece.

    :return: 4 arrays that contains the indices of head/hole for the two categories.
    
    '''

    tot_no_pieces = np.arange(len(side_type))

    # Find which pieces are border pieces
    border_side = np.where(side_type==0)
    border_side_unique = np.unique(border_side[0])

    # Finds which pieces are interior pieces
    interior_pieces_unique = np.setdiff1d(tot_no_pieces,border_side_unique)

    k = 0    
    border_side_type = np.zeros([len(border_side_unique),4],dtype=np.int8)
    for i in border_side_unique:
        border_side_type[k] = side_type[i]
        k += 1

    updated_border_side_type = SortRelevantBorderSide(border_side_type)
    hole_border = np.argwhere(updated_border_side_type == -1).T
    head_border = np.argwhere(updated_border_side_type == 1).T

    # For finding indices of hole/head from border pieces that should be included in interior pieces 
    indices = np.argwhere(border_side_type != updated_border_side_type)
    for i in range(len(indices)):
        indices[i][0] = border_side_unique[indices[i][0]]

    k = 0
    for i,j in zip(hole_border[0],head_border[0]):
           hole_border[0][k] = border_side_unique[i]
           head_border[0][k] = border_side_unique[j]
           k += 1
        
    k = 0
    interior_piece_type = np.zeros([len(interior_pieces_unique),4],dtype=np.int8)
    for i in interior_pieces_unique:
            interior_piece_type[k] = side_type[i]
            k += 1
    hole_interior = np.argwhere(interior_piece_type == -1).T
    head_interior = np.argwhere(interior_piece_type == 1).T

    k = 0
    for i in hole_interior[0]:
        hole_interior[0][k] = interior_pieces_unique[i]
        k += 1
        
    k = 0
    for i in head_interior[0]:
        head_interior[0][k] = interior_pieces_unique[i]
        k += 1
    
    for i,j in indices:
            if side_type[i][j] == -1:
                    hole_interior = np.concatenate((hole_interior,np.array([[i],[j]])),axis = 1)
            elif side_type[i][j] == 1:
                    head_interior = np.concatenate((head_interior,np.array([[i],[j]])),axis = 1)
    
    return hole_interior,head_interior,hole_border,head_border

def matchPieces(side_img,hole,head,threshold):
    '''
    A function that matches combinations of hole and head in the jigsawpuzzle.
    For each matched combination the sides are removed from the list of potential matches and the list is reduced
    until no more combination are left.

    :param side_img: List that contains the side images of head/hole.
    :param hole: An array that contains the index of holes to be matched.
    :param head: An array that contains the index of heads to be matched.

    :return matches: A list containing index of matched hole/head and the corresponding score. 

    '''
    matches = []
    # Initial score check
    i = 0
    while i < len(hole[0]) and len(head[0]) > 0:
        check_score = threshold
        j = 0
        head_idx = 0
        found_match = False
        for k, l in zip(head[0], head[1]):
            if hole[0][i] == k:
                j +=1
                continue
            else:
                score = matchEdges(side_img[hole[0][i]][hole[1][i]], side_img[k][l])
                if score < check_score:
                    check_score = score
                    # indexing score counter
                    head_idx = j
                    found_match = True
                j +=1
        if found_match:    
            matches.append([hole[0,i],
                            hole[1,i],head[0,head_idx],head[1,head_idx],check_score])
            head = np.array([np.delete(head[0], head_idx), np.delete(head[1], head_idx)])
        i += 1
    return matches


def matchPieces_by_dimensions(side_img,hole,head,threshold):
    '''
    A function that matches combinations of hole and head in the jigsawpuzzle.
    For each matched combination the sides are removed from the list of potential matches and the list is reduced
    until no more combination are left.

    :param side_img: List that contains the side images of head/hole.
    :param hole: An array that contains the index of holes to be matched.
    :param head: An array that contains the index of heads to be matched.

    :return matches: A list containing index of matched hole/head and the corresponding score. 

    '''
    matches = []
    # Initial score check
    i = 0
    while i < len(hole[0]) and len(head[0]) > 0:
        check_score = threshold
        j = 0
        head_idx = 0
        found_match = False
        for k, l in zip(head[0], head[1]):
            if hole[0][i] == k:
                j +=1
                continue
            else:
                score = matchDimensions(side_img[hole[0][i]][hole[1][i]], side_img[k][l])
                if score < check_score:
                    check_score = score
                    # indexing score counter
                    head_idx = j
                    found_match = True
                j +=1
        if found_match:    
            matches.append([hole[0,i],
                            hole[1,i],head[0,head_idx],head[1,head_idx],check_score])
            head = np.array([np.delete(head[0], head_idx), np.delete(head[1], head_idx)])
        i += 1
    return matches

def colorThresholder_hsv(img,medianfilter=True):

    '''
    A GUI that allow user to adjust threshold values in the HSV color space using sliders for each parameter in the color space.

    :param img: BGR color image
    :param medianfilter: Applies a median filter to the input image.


    :return: binary mask, segmented image.

    '''

    def nothing(x):
        pass

    # Load image
    img = img.copy()

    # 
    if medianfilter:
        img = cv2.medianBlur(img,9)

    # Create a window to display the image
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640, 480)

    # Create trackbars for HSV thresholds
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    while True:
        # Get current trackbar values
        h_min = cv2.getTrackbarPos('HMin', 'image')
        s_min = cv2.getTrackbarPos('SMin', 'image')
        v_min = cv2.getTrackbarPos('VMin', 'image')
        h_max = cv2.getTrackbarPos('HMax', 'image')
        s_max = cv2.getTrackbarPos('SMax', 'image')
        v_max = cv2.getTrackbarPos('VMax', 'image')

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define HSV color threshold range
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Threshold the HSV image to get only selected colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)

        # Display images
        cv2.imshow('image', res)

        # Check if user has finished adjusting trackbars
        if cv2.waitKey(1) == ord('q'):
            print("Lower array:", lower)
            print("Upper array:", upper)
            cv2.destroyAllWindows()

            return mask,res



def matchDimensions(hole_img,head_img):    

    angle1,angle2,length1,length2,heigth1,height2 = dimensionMatch(hole_img,head_img)
    score = 10*abs(length1-length2) + abs(angle1-angle2) + 10*abs(heigth1[0]-height2[0])

    return score

def dimensionMatch(img_hole,img_head):
    img_head,img_hole = matchEdges(img_hole,img_head,shifted_img=True)
    heigth1 = ConvexDefect(cv2.bitwise_not(img_head))
    heigth2 = ConvexDefect(img_hole)

    img_head = cv2.rotate(img_head,cv2.ROTATE_180)
    img_hole = cv2.bitwise_not(img_hole)
    img_hole = cv2.rotate(img_hole,cv2.ROTATE_180)

    pnt1 = ConvexDefect(img_head,defectpoints=True)
    pnt2 = ConvexDefect(img_hole,defectpoints=True)

    xaxis_pnt1 = abs(pnt1[1][0]-pnt1[0][0])
    yaxis_pnt1 = abs(pnt1[1][1]-pnt1[0][1])

    xaxis_pnt2 = abs(pnt2[1][0]-pnt2[0][0])
    yaxis_pnt2 = abs(pnt2[1][1]-pnt2[0][1])

    angle1 = math.degrees(math.atanh(yaxis_pnt1/xaxis_pnt1))
    angle2 = math.degrees(math.atanh(yaxis_pnt2/xaxis_pnt2))

    distance1 = math.sqrt(xaxis_pnt1**2 + yaxis_pnt1**2)
    distance2 = math.sqrt(xaxis_pnt2**2 + yaxis_pnt2**2)

    return angle1,angle2,distance1,distance2,heigth1,heigth2


def colorThresholder_hsv(img, medianfilter=True, extend_threshold_hue = False):

    '''
    A GUI that allow user to adjust threshold values in the HSV color space using sliders for each parameter in the color space.

    :param img: BGR color image
    :param medianfilter: Applies a median filter to the input image.
    :param extend_threshold_hue: Enables two more sliders for ajusting multiple threshold values for hue.


    :return: binary mask, segmented image.

    '''

    def nothing(x):
        pass

    # Load image
    img = img.copy()
    img = cv2.GaussianBlur(img,(5,5),0.25)
    # 
    if medianfilter:
        img = cv2.medianBlur(img,5)

    # Create a window to display the image
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640, 480)

    # Create trackbars for HSV thresholds
    cv2.createTrackbar('H1Min', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('H1Max', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
    cv2.setTrackbarPos('H1Max', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)
    if extend_threshold_hue:
        cv2.createTrackbar('H2Min', 'image', 0, 179, nothing)
        cv2.createTrackbar('H2Max', 'image', 0, 179, nothing)
        cv2.setTrackbarPos('H2Max', 'image', 179)



    while True:
        # Get current trackbar values
        h_min1 = cv2.getTrackbarPos('H1Min', 'image')
        s_min = cv2.getTrackbarPos('SMin', 'image')
        v_min = cv2.getTrackbarPos('VMin', 'image')
        h_max1 = cv2.getTrackbarPos('H1Max', 'image')
        s_max = cv2.getTrackbarPos('SMax', 'image')
        v_max = cv2.getTrackbarPos('VMax', 'image')
        if extend_threshold_hue:
                h_min2 = cv2.getTrackbarPos('H2Min', 'image')
                h_max2 = cv2.getTrackbarPos('H2Max', 'image')


        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define HSV color threshold range
        lower1 = np.array([h_min1, s_min, v_min])
        upper1 = np.array([h_max1, s_max, v_max])

        if extend_threshold_hue:
            lower2 = np.array([h_min2, s_min, v_min])
            upper2 = np.array([h_max2, s_max, v_max])

            # Threshold the HSV image to get only selected colors
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv,lower2, upper2)
            mask = cv2.bitwise_or(mask1,mask2)
        else:
             mask = cv2.inRange(hsv,lower1,upper1)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)

        # Display images
        cv2.imshow('image', res)

        # Check if user has finished adjusting trackbars
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

            return mask,res
        
def border_frame_assembly(matches,corner_pieces,top_left_corner=0):
    '''
    Finds the sequence of pieces matching together for the whole border frame.
    The algorithm starts by defining the top left corner, this is set by default to be the first corner in the list.
    Then all sides are found by iterating through the match list.

    :param matches: An Nx2 array that contains the the pieces matched together.
    :param corner_pieces: An array contaning the index of all four corners
    :top_left_corner: Used to define which corner should be placed on the top left (not important).

    :return: 4 numpy arrays that contains the sequence pieces for each side of the puzzle.
    
    '''

    solve = True
    next_piece = corner_pieces[top_left_corner]
    previous_piece = corner_pieces[top_left_corner]
    side = []
    all_sides = []
    find_side = True
    a = 0
    sides_found = 0
    while(solve):
        index = np.argwhere(matches == next_piece)
        index = index[:,0]

        if find_side:
            side.append(next_piece)
            mask = matches[index[a]] != next_piece
            previous_piece = next_piece
            next_piece = int(matches[index[a]][mask])
            side.append(next_piece)

        elif np.in1d(previous_piece,matches[index[0]])==False:
            mask = matches[index[0]] != next_piece
            previous_piece = next_piece
            next_piece = int(matches[index[0]][mask])
            side.append(next_piece)

        elif np.in1d(previous_piece,matches[index[1]])==False:
            mask = matches[index[1]] != next_piece
            previous_piece = next_piece
            next_piece = int(matches[index[1]][mask])
            side.append(next_piece)

        find_side = False

        if np.in1d(next_piece, corner_pieces)[0]:
            if sides_found == 0:
                all_sides.append(np.array(side))
                side = []
                sides_found = 1
                a = 1
                next_piece = corner_pieces[0]
                find_side = True     
            elif sides_found == 1:
                all_sides.append(np.array(side))
                side = []
                sequence_of_pieces_found = np.concatenate((all_sides[0],all_sides[1]),axis=None)
                final_corner = np.setdiff1d(corner_pieces,sequence_of_pieces_found)[0]
                next_piece = final_corner
                a = 0
                find_side = True
                sides_found = 2
            elif sides_found == 2:
                all_sides.append(np.array(side))
                side = []
                a = 1
                next_piece = final_corner
                find_side = True
                sides_found = 3
            elif sides_found == 3:
                all_sides.append(np.array(side))
                solve = False
    all_sides = sorted(all_sides,key=lambda x: len(x))
    
    return all_sides    

def tsp_matching(side_img,hole,head,threshold):
    scores = np.zeros((len(hole[0]),len(head[0])))
    for i in range(len(hole[0])):
        for j in range(len(head[0])):
            scores[i,j] = matchEdges(side_img[hole[0][i]][hole[1][i]],side_img[head[0][j]][head[1][j]])
    
    row_ind, col_ind = linear_sum_assignment(scores)
    
    matches = []
    
    for i in range(len(row_ind)):
        if scores[row_ind[i],col_ind[i]] <= threshold:
            matches.append([hole[0,row_ind[i]],hole[1,row_ind[i]],head[0,col_ind[i]],head[1,col_ind[i]],scores[row_ind[i],col_ind[i]]])
            
    return matches

def saveMatchedPairFigure(matches, pieceSides_col):
    # Iterate over each match pair
    for i, match in enumerate(matches):
        # Get the indices of the images in the match pair
        hole_index, hole_side, head_index, head_side, score = match

        # Display the hole image
        plt.figure(figsize=(10, 8))
        plt.subplot(121)
        plt.imshow(pieceSides_col[hole_index][hole_side])
        plt.title('Hole Image')

        # Display the head image
        plt.subplot(122)
        plt.imshow(pieceSides_col[head_index][head_side])
        plt.title('Head Image')

        # Add the score and indices as a text annotation
        text = f'Score: {score}\nHole: [{hole_index}, {hole_side}]\nHead: [{head_index}, {head_side}]'
        plt.text(0.5, 0.05, text, transform=plt.gcf().transFigure,
                 ha='center', fontsize=12)

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save the figure with a unique filename
        filename = f'./image_matches_pairs/match_{i+1}.jpg'
        plt.savefig(filename)
        plt.close()


def tsp_matching2(side_img, hole, head, threshold):
    # This function excludes any head/head or hole/hole matches
    hole_indices = list(range(len(hole[0])))
    head_indices = list(range(len(head[0])))

    scores = np.zeros((len(hole_indices), len(head_indices)))
    for i, hole_idx in enumerate(hole_indices):
        for j, head_idx in enumerate(head_indices):
            try:
                scores[i, j] = matchEdges(side_img[hole[0][hole_idx]][hole[1][hole_idx]], side_img[head[0][head_idx]][head[1][head_idx]])
            except:
                print(hole[0][hole_idx],hole[1][hole_idx],head[0][head_idx],head[1][head_idx])
    row_ind, col_ind = linear_sum_assignment(scores)

    matches = []
    for i in range(len(row_ind)):
        hole_idx = hole_indices[row_ind[i]]
        head_idx = head_indices[col_ind[i]]
        score = scores[row_ind[i], col_ind[i]]
        
        # Exclude head/head and hole/hole combinations
        if hole[0][hole_idx] == head[0][head_idx]:
            continue
        
        matches.append([hole[0][hole_idx], hole[1][hole_idx], head[0][head_idx], head[1][head_idx], score])

    return matches
