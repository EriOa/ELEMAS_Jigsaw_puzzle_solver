import numpy as np
from functions import *
import matplotlib.pyplot as plt

def rotation(degree):
    if degree == 90 or degree == -270:
        rot = cv2.ROTATE_90_CLOCKWISE
    elif degree == -90 or degree == 270:
        rot = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif degree == 180 or degree == -180:
        rot = cv2.ROTATE_180
    else:
        rot = None
    return rot 

def correct_rotation_interior(img,org_orientation_of_fit, corrected_orientation):
    img_rot = img.copy()
    total_rotation = corrected_orientation
    if corrected_orientation != 0:
        img_rot = cv2.rotate(img_rot,rotation(corrected_orientation))
    
    if org_orientation_of_fit == 0:
        img_rot = cv2.rotate(img_rot,rotation(90))
        total_rotation += -90
    elif org_orientation_of_fit == 1:
        img_rot = cv2.rotate(img_rot,rotation(-90))
        total_rotation += 90
    elif org_orientation_of_fit == 2:
        img_rot = cv2.rotate(img_rot,rotation(90))
        total_rotation += -90
    elif org_orientation_of_fit == 3:
        img_rot = cv2.rotate(img_rot,rotation(-90))
        total_rotation += -180

    
    return img_rot,total_rotation
    
    
def correct_rotation_border(type,img,rotate='TopLeft'):
    '''
    A method for rotation of border pieces. The function rotates to the standard "TopLeft"- configuration.

    :param type: An array with the type description of the puzzle piece
    :param img: Image of the puzzle piece
    :param rotate: Indicates what type of side the rotation is to perform:
                    TopLeft:        Rotates to the upper frame and if piece is a corner it will be in the top left
                    TopRight:       Rotates to the right frame and if piece is a corner it will be in the top right
                    BottomLeft:     Rotates to the left frame and if piece is a corner it will be in the bottom left
                    BottomRight:    Rotates to the bottom frame and if piece is a corner it will be in the bottom right
                                    
    :return: rotated image and an integer number with the amount of rotation performed                                    
    '''

    rot_c90 = cv2.ROTATE_90_CLOCKWISE
    rot_cc90 = cv2.ROTATE_90_COUNTERCLOCKWISE
    rot_180 = cv2.ROTATE_180
    type = type.copy()
    img = img.copy()
    # To keep track of rotations
    rotation = 0
    # Find number of sides to indicate cornerpieces
    no_sides = len(np.argwhere(type==0))

    if no_sides == 2:
        if type[2] == 0 and type[3] == 0:
            img = cv2.rotate(img,rot_180)
            rotation += -180
        elif type[1] == 0 and type[3] == 0:
            img = cv2.rotate(img,rot_c90)
            rotation += 90
        elif type[0] == 0 and type[2] == 0:
            img = cv2.rotate(img,rot_cc90)
            rotation -= 90

    elif no_sides == 1:
        if type[1] == 0:
            img = cv2.rotate(img,rot_c90)
            rotation += 90
        elif type[2] == 0:
            img = cv2.rotate(img,rot_cc90)
            rotation -= 90
        elif type[3] == 0:
            img = cv2.rotate(img,rot_180)
            rotation += 180
    else:
        print('The following piece is not a sidepiece')

    if rotate == 'TopRight':
        img = cv2.rotate(img,rot_c90)
        rotation += 90
    elif rotate == 'BottomLeft':
        img = cv2.rotate(img,rot_cc90)
        rotation -= 90
    elif rotate == 'BottomRight':
        img = cv2.rotate(img, rot_180)
        rotation += 180
        

    return img, rotation

def interior_assembly(row_to_compare,column_to_compare, interior_matches):
    ''' 
    Sorts interior pieces into the column-wise sequence they appear in the solution.

    :param row_to_compare: The upper row sequence of the border pieces
    :param column_to_compare: The left side column sequence of the border pieces.

    :return: A list containing an array for each column sequence in the interior pieces.    
    '''

    sequence_of_pieces = []
    for k in range(len(row_to_compare)):
        column = np.zeros(len(column_to_compare),dtype=np.uint8)
        new_column = np.zeros(len(column_to_compare),dtype=np.uint8)
        for i,j in enumerate(column_to_compare):
            check_on_first_column = np.argwhere(interior_matches[:,0]==j)
            check_on_second_column = np.argwhere(interior_matches[:,2]==j)
            if check_on_first_column.size>0:
                column[i] = int(check_on_first_column)
                if interior_matches[column[i],0] != j:
                    new_column[i] = interior_matches[column[i],0]
                else:
                    new_column[i] = interior_matches[column[i],2]
            elif check_on_second_column.size>0:
                column[i] = int(check_on_second_column)
                if interior_matches[column[i],0] != j:
                    new_column[i] = interior_matches[column[i],0]
                else:
                    new_column[i] = interior_matches[column[i],2]
        interior_matches = np.delete(interior_matches,column,0)
        sequence_of_pieces.append(new_column)
        column_to_compare = new_column

        # The sequence is calculated based on the previous column and thus matches in same column are removed
        for i in range(len(new_column[0:-1])):
            not_relevant1 = np.argwhere((interior_matches[:,0]==new_column[i]) & (interior_matches[:,2]==new_column[i+1]))
            not_relevant2 = np.argwhere((interior_matches[:,0]==new_column[i+1]) & (interior_matches[:,2]==new_column[i]))
            if not_relevant1.size>0:
                interior_matches = np.delete(interior_matches,int(not_relevant1),0)
            elif not_relevant2.size>0:
                interior_matches = np.delete(interior_matches,int(not_relevant2),0)

                
    return sequence_of_pieces

def flip_side_type(side_type):
    side_type = side_type.copy()
    side_flip = np.array([1, 0, -1, 0])
    side_flip2 = np.array([-1, 0, 1, -1])
    side_flip3 = np.array([-1, 1, 0, -1])
    side_flip4 = np.array([-1, 1, 0, 0])
    side_flip5 = np.array([1, -1, 0, 1])
    side_flip6 = np.array([0, -1, 0, 1])
    
    # Skriv om denne!! er ikke nødvendig å skrive alle typer kombinasjoner, bare flip 1 og 2.. 

    for i in range(len(side_type)):
        if np.array_equal(side_type[i],side_flip):
            side_type[i] = np.array([1, -1, 0, 0])
        elif np.array_equal(side_type[i],side_flip2):
            side_type[i] = np.array([-1, 1, 0, -1])
        elif np.array_equal(side_type[i], side_flip3):
            side_type[i] = np.array([-1, 0, 1, -1])
        elif np.array_equal(side_type[i],side_flip4):
            side_type[i] = np.array([-1, 0, 1, 0])
        elif np.array_equal(side_type[i],side_flip5):
            side_type[i] = np.array([1, 0, -1, 1])
        elif np.array_equal(side_type[i],side_flip6):
            side_type[i] = np.array([0, 0, -1, 1])
    
    return side_type

def visualize_puzzle_solver(img,side_type,match_border,match_interior,size=200,grid_return=False):
    
    '''
    :param img: list of binary images of each single piece in the puzzle
    :param side_type: The configuration of edges
    :param match_border: List containing matched border edges
    :param match_interior: List containing matched interior pieces
    :param size: Size of the rectangle images (nxn)
    
    
    :return grid_return: Returns the final image in a grid arrangement 
    
    '''

    ################################
    ####### Border Assembly ########
    ################################

    # Identify corner pieces
    border = np.where(side_type==0)
    border_unique,counts = np.unique(border[0],return_counts=True)
    corner_pieces = border_unique[counts>1]

    # Resize to square images and flip
    resized_images_org = []
    for i in range(len(img)):
        flipp_img = cv2.flip(img[i],1)
        resized_images_org.append(cv2.resize(flipp_img,(size,size)))
    
    # For flipping the side type also
    side_type2 = flip_side_type(side_type)
    
    resized_images = resized_images_org.copy()
    
    # For only getting piece index of matched pieces
    match2 = np.zeros((len(match_border),2),dtype=np.int8)
    for i in range(len(match_border)):
        match2[i] = ([match_border[i][0],match_border[i][2]])

    sides = border_frame_assembly(match2,corner_pieces)
    
    # Correct the orietation of border pieces
    rotation_pieces = []
    top_images = []
    bottom_images = []
    for i in range(len(sides[0])-1):
         piece_nr = sides[0][i]
         next_image, rotation = correct_rotation_border(side_type2[piece_nr],resized_images[piece_nr])
         top_images.append(next_image)
         rotation_pieces.append([piece_nr,rotation])
         piece_nr = sides[1][i]
         next_image, rotation = correct_rotation_border(side_type2[piece_nr],resized_images[piece_nr],rotate='BottomRight')
         bottom_images.append(next_image)
         rotation_pieces.append([piece_nr,rotation])

    left_images = []
    right_images = []
    for i in range(len(sides[2])-1):
        piece_nr = sides[2][i+1]
        next_image, rotation = correct_rotation_border(side_type2[piece_nr],resized_images[piece_nr],rotate='BottomLeft')
        left_images.append(next_image)
        rotation_pieces.append([piece_nr,rotation])
        piece_nr = sides[3][i+1]
        next_image, rotation = correct_rotation_border(side_type2[piece_nr],resized_images[piece_nr],rotate='TopRight')
        right_images.append(next_image)
        rotation_pieces.append([piece_nr,rotation])

    column = len(sides[0])
    rows = len(sides[2])
    # Create an empty canvas
    grid = np.zeros((rows*size,column*size,3),dtype=np.uint8)

    # Places the images in a grid structure
    for i in range(len(top_images)):
        grid[0:size,i*size:(i+1)*size] = top_images[i]
        grid[(rows-1)*size:rows*size,(column-1-i)*size:(column-i)*size] = bottom_images[i]
    for i in range(len(right_images)):
        grid[(rows-1-i)*size:(rows-i)*size,0:size] = left_images[column-i]
        grid[i*size:(i+1)*size,(column-1)*size:column*size] = right_images[len(left_images)-i-1]
        
    
    ##################################
    ####### Interior Assembly ########
    ##################################

    match_arr = np.array(match_interior)
    rot_arr = np.array(rotation_pieces)

    # Removes unrelevant matches in interior
    for i in sides[0]:
        check1 = np.argwhere(match_arr[:,0]==i)
        check2 = np.argwhere(match_arr[:,2]==i)

        if check1.size>0:
            match_arr = np.delete(match_arr,int(check1),0)
        if check2.size>0:
            match_arr = np.delete(match_arr,int(check2),0)
            
    for i in sides[1]:
        check1 = np.argwhere(match_arr[:,0]==i)
        check2 = np.argwhere(match_arr[:,2]==i)
        if check1.size>0:
            match_arr = np.delete(match_arr,int(check1),0)
        if check2.size>0:
            match_arr = np.delete(match_arr,int(check2),0)
    
    # Finds the sequence for each column in the interior pieces
    interior_sequence = interior_assembly(sides[0][1:-1],sides[2][1:-1],match_arr)
    resized_images[13] = cv2.rotate(resized_images[13],cv2.ROTATE_90_CLOCKWISE)
    left_column_to_matches = sides[2][1:-1]
    for i in range(len(interior_sequence)):
        new_column = []
        piece_and_rotation = np.zeros((len(interior_sequence[0]),2), dtype=np.int64)
        for j in range(len(interior_sequence[0])):
            piece_idx = interior_sequence[i][j]
            condition1 = np.isin(left_column_to_matches[j], match_arr[:, 0])
            condition2 = np.isin(piece_idx, match_arr[:, 2])
            condition3 = np.isin(left_column_to_matches[j], match_arr[:, 2])
            condition4 = np.isin(piece_idx, match_arr[:, 0])

            # Check if all conditions are true and indices match
            if np.all(condition1 & condition2) and (np.argwhere((match_arr[:, 0] == left_column_to_matches[j]) & (match_arr[:, 2] == piece_idx)).size!=0):
                indices = np.argwhere((match_arr[:, 0] == left_column_to_matches[j]) & (match_arr[:, 2] == piece_idx))
                index_of_rot = np.argwhere(rot_arr[:,0]==left_column_to_matches[j])
                resized_images[piece_idx],tot_rot = correct_rotation_interior(resized_images[piece_idx],int(match_arr[(indices[0]), 1]),rot_arr[index_of_rot,1])
                new_column.append(match_arr[int(indices[0]), 2])
                piece_and_rotation[j] = np.array([int(match_arr[int(indices[0]), 2]), int(tot_rot)])
            elif np.all(condition3 & condition4) and (np.argwhere((match_arr[:, 2] == left_column_to_matches[j]) & (match_arr[:, 0] == piece_idx)).size!=0):
                index_of_rot = np.argwhere(rot_arr[:,0]==left_column_to_matches[j])
                indices = np.argwhere((match_arr[:, 2] == left_column_to_matches[j]) & (match_arr[:, 0] == piece_idx))
                resized_images[piece_idx],tot_rot = correct_rotation_interior(resized_images[piece_idx],int(match_arr[(indices[0]), 3]),rot_arr[index_of_rot,1])
                new_column.append(match_arr[int(indices[0]), 0])
                piece_and_rotation[j] = np.array([int(match_arr[int(indices[0]), 0]), int(tot_rot)])
        
        left_column_to_matches = np.array(new_column)
        rot_arr = piece_and_rotation
    
    # Place the pieces on the grid
    for i in range(len(interior_sequence)):
        for j in range(interior_sequence[i].size):
            piece_index = interior_sequence[i][j]
            grid[(j+1)*size:(j+2)*size, (i+1)*size:(i+2)*size] = resized_images[piece_index]
            

    return grid

