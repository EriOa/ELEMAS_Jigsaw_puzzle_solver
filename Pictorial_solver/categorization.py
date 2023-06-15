import cv2
from scipy.ndimage import maximum_filter, minimum_filter
#from scipy.spatial.distance import pdist, squareform
import math
import numpy as np
import matplotlib.pyplot as plt
from functions import BinaryMask

def load_img(num):
    bin_img_lst = []
    img_lst = []
    for i in range(num):
        path1 = '.\Cropped_Image2/Cropped_ImageCroppedImage{}.jpg'.format(i+1)
        path2 = '.\Cropped_Bin_Image/Cropped_ImageCroppedImage{}.jpg'.format(i+1)
        dst = cv2.imread(path2)
        gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        _, Bin_img1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        bin_img_lst.append(Bin_img1)
        col_BGR = cv2.imread(path1)
        col_RGB = cv2.cvtColor(col_BGR,cv2.COLOR_BGR2RGB)
        img_lst.append(col_RGB)
    
    return bin_img_lst,img_lst

def cart2pol(x, y):
    der = []
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    deri_r = np.tan(phi) * np.sqrt((x ** 2) / np.cos(phi) ** 2)
    if deri_r == 0:
        # circle = plt.Circle((phi,rho),2, color='g')
        # plt.gca().add_patch(circle)
        der.append([phi, rho])

    return (rho, phi)


def pol2cart(A1, A2, cx, cy):
    x = A2 * np.cos(A1)
    y = A2 * np.sin(A1)
    x = int(x + cx)
    y = int(y + cy)
    return x, y


def Contourplot(imagename, dst):
    # The input of method is a image and it will find out corner location from the image.
    # Find contours on after binary mask image

    image_contour = imagename.copy()

    w, h, _ = image_contour.shape
    orig_x = int(w / 2)
    orig_y = int(h / 2)

    # Use non approx or it will miss some point on the plot
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(orig_x, orig_y)
    longest_contour = None
    max_contour_length = 0
    for contour in contours:
        contour_length = cv2.arcLength(contour, True)
        if contour_length > max_contour_length:
            longest_contour = contour
            max_contour_length = contour_length

    # convert contour points to integers
    longest_contour = np.squeeze(np.int32(longest_contour))
    # print('contours', contours)
    # cv2.drawContours(image_contour, contours, -1, (0, 0, 255), 3)  # Draw contours on original image
    # get contour points in polar coordinates
    M = cv2.moments(longest_contour)
    # Find centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # print('cx, cy', cx, cy)
    polar = [cart2pol(c[0] - cx, c[1] - cy) for c in longest_contour]
    max_i = polar.index(max(polar, key=lambda x: x[1]))
    polar = polar[max_i:] + polar[:max_i]
    ds, phis = zip(*polar)

    A1 = np.array(phis)
    A2 = np.array(ds)

    # plt.plot(A1, A2)
    # plt.show()
    ############## need to create a new function to this part ##################
    relevant_peak_indices = []
    size = 300
    max_filtered = maximum_filter(A2, size)
    min_filtered = minimum_filter(A2, size)
    peak_indices = np.where(max_filtered == A2)[0]
    valley_indices = np.array(np.where(min_filtered == A2)[0])
    # print(peak_indices)

    mask = (abs(A1) < 2.8) & (abs(A1) > 2.0)  # create a boolean mask for the desired interval
    relevant_indices = np.where(mask)[0]  # get the indices of the elements that satisfy the condition
    relevant_peak_indices1 = np.intersect1d(peak_indices, relevant_indices)  # find the intersection with peak_indices

    # [plt.scatter(A1[i], A2[i], color='r') for i in valley_indices]  # plot the relevant points
    # [plt.scatter(A1[i], A2[i], color='g') for i in relevant_peak_indices1]  # plot the relevant points
    # Calculate and plot the corresponding cartesian points
    # a = np.array(peak_indices + valley_indices.flatten())
    # relevant_peak_indices.append(relevant_peak_indices1[0])
    merged_array = np.concatenate((peak_indices, valley_indices))
    for i in relevant_peak_indices1:
        # plt.scatter(A1[i], A2[i], color='g')
        # print(i, A1[i], A2[i])
        relevant_peak_indices.append(i)

    mask = (abs(A1) > 0.5) & (abs(A1) < 1.1)  # Create a boolean mask for the desired interval
    relevant_indices = np.where(mask)[0]  # get the indices of the elements that satisfy the condition
    relevant_peak_indices1 = np.intersect1d(peak_indices,
                                            relevant_indices)  # get the indices of the elements that satisfy.
    # [plt.scatter(A1[i], A2[i], color='g') for i in relevant_peak_indices1]
    # relevant_peak_indices.append(relevant_peak_indices1[0])
    for i in relevant_peak_indices1:
        # plt.scatter(A1[i], A2[i], color='g')
        # print(i, A1[i], A2[i])
        relevant_peak_indices.append(i)

    i = 0
    threshold = 20
    while i < len(relevant_peak_indices) - 1:
        if abs(relevant_peak_indices[i + 1] - relevant_peak_indices[i]) <= threshold:
            average = int((relevant_peak_indices[i + 1] + relevant_peak_indices[i]) / 2)
            relevant_peak_indices.pop(i)
            relevant_peak_indices.pop(i)
            relevant_peak_indices.insert(i, average)
        else:
            i += 1

    # print(relevant_peak_indices)
    corners = []

    for i in relevant_peak_indices:
        x, y = pol2cart(A1[i], A2[i], cx, cy)
        # Add the circle to the axis object
        corners.append([x, y])

    corners = CornerOrder(corners, orig_y, orig_x)

    points = contours[0][:, 0, :]
 

    x = points[:, 0]
    y = points[:, 1]

    return corners


def CornerOrder(corners, orig_x, orig_y):
    n_cnr = [[coord[0] - orig_x, orig_y - coord[1]] for coord in corners]

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

    return res


def rotationAboutFixedAxis(img, pointOfRotation, secondPoint):
    """
    Calculates the angle of rotation and rotates the image so that the two points lie on the same y-axis.

    :param img: Image
    :param pointOfRotation: Specify point of rotation in tuple(int(x),int(y)) coordinates.
    :param secondPoint: Point to be rotated (x,y)

    :return: Rotated image

    """

    angle = math.atan((secondPoint[1] - pointOfRotation[1]) / (secondPoint[0] - pointOfRotation[0])) * 180 / math.pi
    M = cv2.getRotationMatrix2D(pointOfRotation, angle, 1)
    dst = cv2.warpAffine(img, M, (0, 0))

    return dst


def edgeFeatureExtraction(img):
    """
    Function that finds the edge contours in an image and
    returns a cropped image with only the dominant edge

    :param img: Binary image of cropped side piece

    :return: image containing only the longest edge.
    
    """

    # Canny edge detection
    edge = cv2.Canny(img, threshold1=200, threshold2=700)

    # Find contours from canny edge
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # If there are nothing in contours
    if len(contours) != 0:
        top_contour = [len(i) for i in contours]

        index = np.argmax(np.array(top_contour))

        for i in range(len(contours)):
            if i == index:
                pass
            else:
                cv2.boundingRect(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                img[y:y + h, x:x + w] = 255

        return img
    else:
        return img
    
    
def ConPlot_CropImg_sides(img, bin_img, ylim_min, ylim_max, mask=False, plot=False, gray_plot=False, median = 27, morph=(7,7)):
    # hsv = binarymask_HSV(img)
    cnr = Contourplot(img, bin_img)
    img_rot = rotationAboutFixedAxis(img, (int(cnr[0][1]), int(cnr[0][1])), cnr[1])
    img_rot2 = rotationAboutFixedAxis(bin_img, (int(cnr[0][1]), int(cnr[0][1])), cnr[1])
    cnr1 = Contourplot(img_rot, img_rot2)
    # print(cnr1)
    topSide_bin = img_rot2[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]]
    topSide_col = img_rot[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]]

    img90c_col = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img90c_bin = cv2.rotate(bin_img, cv2.ROTATE_90_CLOCKWISE)
    cnr = Contourplot(img90c_col, img90c_bin)
    img_rot_90c_bin = rotationAboutFixedAxis(img90c_bin, (int(cnr[0][1]), int(cnr[0][1])), cnr[1])
    img_rot_90c_col = rotationAboutFixedAxis(img90c_col, (int(cnr[0][1]), int(cnr[0][1])), cnr[1])
    cnr2 = Contourplot(img_rot_90c_col, img_rot_90c_bin)
    leftSide_bin = img_rot_90c_bin[ylim_min:ylim_max, (cnr2[0][0]):cnr2[1][0]]
    leftSide_col = img_rot_90c_col[ylim_min:ylim_max, (cnr2[0][0]):cnr2[1][0]]

    img180c_col = cv2.rotate(img90c_col, cv2.ROTATE_90_CLOCKWISE)
    img180c_bin = cv2.rotate(img90c_bin, cv2.ROTATE_90_CLOCKWISE)
    cnr = Contourplot(img180c_col, img180c_bin)
    img_rot180c_col = rotationAboutFixedAxis(img180c_col, (int(cnr[0][0]), int(cnr[0][1])), cnr[1])
    img_rot180c_bin = rotationAboutFixedAxis(img180c_bin, (int(cnr[0][0]), int(cnr[0][1])), cnr[1])
    cnr3 = Contourplot(img_rot180c_col, img180c_bin)
    bottomSide_col = img_rot180c_col[ylim_min:ylim_max, (cnr3[0][0]):cnr3[1][0]]
    bottomSide_bin = img_rot180c_bin[ylim_min:ylim_max, (cnr3[0][0]):cnr3[1][0]]

    img270c_col = cv2.rotate(img180c_col, cv2.ROTATE_90_CLOCKWISE)
    img270c_bin = cv2.rotate(img180c_bin, cv2.ROTATE_90_CLOCKWISE)
    cnr = Contourplot(img270c_col, img270c_bin)
    img_rot270c_col = rotationAboutFixedAxis(img270c_col, (int(cnr[0][0]), int(cnr[0][1])), cnr[1])
    img_rot270c_bin = rotationAboutFixedAxis(img270c_bin, (int(cnr[0][0]), int(cnr[0][1])), cnr[1])
    cnr4 = Contourplot(img_rot270c_col, img_rot270c_bin)
    rightSide_col = img_rot270c_col[ylim_min:ylim_max, (cnr4[0][0]):cnr4[1][0]]
    rightSide_bin = img_rot270c_bin[ylim_min:ylim_max, (cnr4[0][0]):cnr4[1][0]]

    
    topSide_bin = cv2.medianBlur(topSide_bin, median)
    leftSide_bin = cv2.medianBlur(leftSide_bin, median)
    rightSide_bin = cv2.medianBlur(rightSide_bin, median)
    bottomSide_bin = cv2.medianBlur(bottomSide_bin, median)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph)
    topSide_bin = cv2.dilate(topSide_bin, kernel, iterations=1)
    topSide_bin = cv2.erode(topSide_bin, kernel, iterations=1)
    leftSide_bin = cv2.dilate(leftSide_bin, kernel, iterations=1)


    
    leftSide_bin = cv2.erode(leftSide_bin, kernel, iterations=1)
    rightSide_bin = cv2.dilate(rightSide_bin, kernel, iterations=1)
    rightSide_bin = cv2.erode(rightSide_bin, kernel, iterations=1)
    bottomSide_bin = cv2.dilate(bottomSide_bin, kernel, iterations=1)
    bottomSide_bin = cv2.erode(bottomSide_bin, kernel, iterations=1)
    topSide_bin = BinaryMask(topSide_bin)
    leftSide_bin = BinaryMask(leftSide_bin)
    rightSide_bin = BinaryMask(rightSide_bin)
    bottomSide_bin = BinaryMask(bottomSide_bin)
    
    if mask == True:
        topSide_bin = edgeFeatureExtraction(topSide_bin)
        leftSide_bin = edgeFeatureExtraction(leftSide_bin)
        rightSide_bin = edgeFeatureExtraction(rightSide_bin)
        bottomSide_bin = edgeFeatureExtraction(bottomSide_bin)
    if plot == True:
        plt.figure(figsize=(8, 5))
        plt.subplot(221)
        plt.title('Topside')
        plt.imshow(topSide_col, cmap='gray', vmin=0, vmax=255)
        plt.subplot(222)
        plt.title('Leftside')
        plt.imshow(leftSide_col, cmap='gray', vmin=0, vmax=255)
        plt.subplot(223)
        plt.title('Rightside')
        plt.imshow(rightSide_col, cmap='gray', vmin=0, vmax=255)
        plt.subplot(224)
        plt.title('Bottomside')
        plt.imshow(bottomSide_col, cmap='gray', vmin=0, vmax=255)
        plt.show()
    if gray_plot == True:
        # topSide_gray = img_rot2[ylim_min:ylim_max, cnr1[0][0]:cnr1[1][0]]
        # leftSide_gray = img_rot_90c_bin[ylim_min:ylim_max, (cnr2[0][0]):cnr2[1][0]]
        # bottomSide_gray = img_rot180c[ylim_min:ylim_max, cnr3[0][0]:cnr3[1][0]]
        # rightSide_gray = img_rot270c[ylim_min:ylim_max, (cnr4[0][0]):cnr4[1][0]]
        plt.figure(figsize=(8, 5))
        plt.subplot(221)
        plt.title('Topside')
        plt.imshow(topSide_bin, cmap='gray', vmin=0, vmax=255)
        plt.subplot(222)
        plt.title('Leftside')
        plt.imshow(leftSide_bin, cmap='gray', vmin=0, vmax=255)
        plt.subplot(223)
        plt.title('Rightside')
        plt.imshow(rightSide_bin, cmap='gray', vmin=0, vmax=255)
        plt.subplot(224)
        plt.title('Bottomside')
        plt.imshow(bottomSide_bin, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return [topSide_bin, leftSide_bin, rightSide_bin, bottomSide_bin], [topSide_col, leftSide_col, rightSide_col,
                                                                        bottomSide_col]
    
    
def SideType(img):
    '''
    For checking what type of side the img belongs to

    :param: Cropped image of one side of puzzle piece

    :return: Integer from 0 - 2 depending on which type of sidetype it belongs to.
    '''

    defects = ConvexDefect(img)
    if len(defects) == 0:
        return int(0)
    elif len(defects) == 1:
        return int(-1)
    elif len(defects) == 2:
        return int(1)


def ConvexDefect(img):
    '''
    :param img: Binary image
    :return: Color image with defect and convex
    '''

    imagename = img.copy()
    contours, hierarchy = cv2.findContours(imagename, 2, 1)

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt = sorted_contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    defs = []
    length_from_hull = []
    for i in range(defects.shape[0]):
        if defects[i][0][3] > 5000:
            defs.append(defects)
            s, e, f, d, = defects[i, 0]
            far = tuple(cnt[f][0])
            cv2.circle(imagename, far, 5, [0, 0, 255], -1)
            length_from_hull.append(defects[i][0][3] / 256)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])

        cv2.line(imagename, start, end, [0, 255, 255], 2)

        # print('far', start)

    # plt.imshow(imagename,cmap='gray')
    return length_from_hull

def readPieceSides_col(col_img_array, bin_img_array):
    num_images = len(col_img_array)
    pieceSides_bin = [None] * num_images
    pieceSides_col = [None] * num_images
    piecetype = np.zeros((num_images, 4),dtype=int)

    for i in range(num_images):
        bin_cropped, col_cropped = ConPlot_CropImg_sides(col_img_array[i], bin_img_array[i], 0, 340, True, False, False,median=27,morph=(7,7))
        pieceSides_bin[i] = bin_cropped
        pieceSides_col[i] = col_cropped
        piecetype[i] = (
            SideType(bin_cropped[0]), SideType(bin_cropped[1]), SideType(bin_cropped[2]), SideType(bin_cropped[3]))

    return pieceSides_bin, pieceSides_col, piecetype
