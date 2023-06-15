import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def colour_sampling_visual(binary_side,
                    colour_side,
                    num_samples=50,
                    rect_size=8,
                    min_distance=20,
                    plot_inner_contour=False):

    '''
    For taking colour samples along the shrunken contour edge. 
    
    '''
    binary_img = binary_side.copy()
    colour_img = colour_side.copy()

    # Applying morphological operation to shrink the contour
    binary_img = cv2.GaussianBlur(binary_img, (13, 13), 0)
    edge = cv2.Canny(binary_img, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    edge_dilate = cv2.dilate(edge, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode = cv2.erode(edge_dilate, kernel, iterations=1)
    inner_edge_mask = edge_dilate - erode

    contours, _ = cv2.findContours(inner_edge_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if plot_inner_contour:
        inner_contour = cv2.drawContours(edge, contours, 0, color=(255, 255, 255))
        plt.imshow(inner_contour)

    contour_points = contours[0][:, 0, :]
    num_contour_points = len(contour_points)
    interval = num_contour_points // num_samples

    samples_window = []
    colours = []
    sampled_points = []
    samples = []

    for i in range(0, num_contour_points, interval):
        pt = tuple(map(int, contour_points[i]))

        if len(sampled_points) > 0:
            # calculate distance to previously sampled points
            distances = np.linalg.norm(np.array(pt) - np.array(sampled_points), axis=1)
            if np.any(distances < min_distance):
                continue

        # calculate offset based on distance
        offset = rect_size // 2

        pt1 = (int(pt[0] - offset), int(pt[1] - offset))
        pt2 = (int(pt[0] + offset), int(pt[1] + offset))
        if pt1[0] <= 10 or pt2[0] >= colour_img.shape[1]-10:
            continue
        cv2.rectangle(binary_img, pt1, pt2, (0, 255, 0), 2)

        if i == 0:
            continue

        rect_sample = colour_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        rect_sample = cv2.cvtColor(rect_sample,cv2.COLOR_RGB2HSV)
        if np.any(rect_sample):
            samples_window.append(rect_sample)
            colours.append(rect_sample.mean(axis=(0, 1)))
            
        samples.append(np.array([pt1[0], pt1[1], pt2[0], pt2[1]], dtype=np.int32))
        sampled_points.append(pt)

    return samples_window, colours, samples


def calcScore(hole_bin, hole_col, head_bin, head_col, alpha=290, beta=340, threshold=15):
    '''
    Calculates the score of a hole/head combination
    
    :param hole_bin: Binary edge image of a hole
    :param hole_col: Colour image of a head corresponding to the Binary image
    :param head_bin: Binary edge image of a hole
    :param head_col: Colour image of a head corresponding to the Binary image
    :param alpha: Weigthing factor of the overlay local matching procedure
    :param beta:  Weigthing factor of the colour local matching procedure
    :param threshold: Minimun corner distance threshold
    
    :Return: Similarity score (float)
    
    '''
    
    if np.sqrt(np.sum(np.power(hole_bin.shape[1] - head_bin.shape[1], 2))) > threshold:
        score = 10000
    else:
        overlay_score = matchEdges(hole_bin, head_bin)
        normalized_overlay = overlay_score / 10000
        normalized_overlay = min(normalized_overlay, 1)
        score = alpha * normalized_overlay
        
        if beta != 0:
            _, colour_values_hole, _ = colour_sampling_visual(hole_bin, hole_col)
            _, colour_values_head, _ = colour_sampling_visual(head_bin, head_col)
            
            mean_colour_hole = np.mean(colour_values_hole, axis=0)
            mean_colour_head = np.mean(colour_values_head, axis=0)
            
            colour_diff = np.sqrt(np.sum(np.power(mean_colour_hole - mean_colour_head, 2)))
            colour_diff_normalized = colour_diff / 403
            score += beta * colour_diff_normalized
    
    return score

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
    
def colour_mathing_LAP(side_img_bin, side_img_col, hole, head, threshold=10000):
    """
    LAP Global matching method
    
    :param side_img_bin: list of binary side images
    :param side_img_col: list of coloured side images
    :param hole: Configuration of hole list
    :param head: Configuration of head list
    
    :return: List containing matched edges 
    
    """
    
    
    
    num_holes = len(hole[0])
    num_heads = len(head[0])
    
    scores = np.zeros((num_holes, num_heads))

    for i in range(num_holes):
        hole_bin = side_img_bin[hole[0][i]][hole[1][i]]
        hole_col = side_img_col[hole[0][i]][hole[1][i]]

        for j in range(num_heads):
            head_bin = side_img_bin[head[0][j]][head[1][j]]
            head_col = side_img_col[head[0][j]][head[1][j]]

            scores[i, j] = calcScore(hole_bin, hole_col, head_bin, head_col,threshold=15)

    row_ind, col_ind = linear_sum_assignment(scores)

    matches = []

    for i in range(len(row_ind)):
        score = scores[row_ind[i], col_ind[i]]
        if score <= threshold:
            hole_idx = [hole[0][row_ind[i]], hole[1][row_ind[i]]]
            head_idx = [head[0][col_ind[i]], head[1][col_ind[i]]]
            matches.append(hole_idx + head_idx + [score])

    return matches


