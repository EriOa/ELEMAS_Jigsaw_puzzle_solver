import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def correct_rotation_interior(img,org_orientation_of_fit):
    """
    Corrects the rotaion of the interior pieces. Once the frame is complete the images are placed column-wise
    starting from the left side of the frame.
    
    :param img: colour image
    :org_orientation_of fit: The side integer(0-3) in which the piece was originally fit to the left column piece
    according to the match list.
    
    :return img_rot: Corrects the rotation so that image is in correspondance to the previous column image.
    
    """
    
    img_rot = img.copy()
    if org_orientation_of_fit == 0:
        img_rot = cv2.rotate(img_rot,rotation(-90))
    elif org_orientation_of_fit == 1:
        img_rot = img_rot
    elif org_orientation_of_fit == 2:
        img_rot = cv2.rotate(img_rot,rotation(180))
    elif org_orientation_of_fit == 3:
        img_rot = cv2.rotate(img_rot,rotation(90))
    
    return img_rot

def correct_rotation_border(types,img,rotate='TopLeft'):
    '''
    A method for rotation of border pieces. The function rotates to the standard "TopLeft"- configuration.

    :param types: An array with the types description of the puzzle piece
    :param img: Image of the puzzle piece
    :param rotate: Indicates what types of side the rotation is to perform:
                    TopLeft:        Rotates to the upper frame and if piece is a corner it will be in the top left
                    TopRight:       Rotates to the right frame and if piece is a corner it will be in the top right
                    BottomLeft:     Rotates to the left frame and if piece is a corner it will be in the bottom left
                    BottomRight:    Rotates to the bottom frame and if piece is a corner it will be in the bottom right
                                    
    :return: rotated image
                                        
    '''

    rot_c90 = cv2.ROTATE_90_CLOCKWISE
    rot_cc90 = cv2.ROTATE_90_COUNTERCLOCKWISE
    rot_180 = cv2.ROTATE_180
    types = types.copy()
    img = img.copy()

    # Find number of sides to indicate cornerpieces
    no_sides = len(np.argwhere(types==0))

    if no_sides == 2:
        if types[2] == 0 and types[3] == 0:
            img = cv2.rotate(img,rot_180)
        elif types[1] == 0 and types[3] == 0:
            img = cv2.rotate(img,rot_c90)
        elif types[0] == 0 and types[2] == 0:
            img = cv2.rotate(img,rot_cc90)

    elif no_sides == 1:
        if types[1] == 0:
            img = cv2.rotate(img,rot_c90)
        elif types[2] == 0:
            img = cv2.rotate(img,rot_cc90)
        elif types[3] == 0:
            img = cv2.rotate(img,rot_180)
    else:
        print('The following piece is not a sidepiece')

    if rotate == 'TopRight':
        img = cv2.rotate(img,rot_c90)
    elif rotate == 'BottomLeft':
        img = cv2.rotate(img,rot_cc90)
    elif rotate == 'BottomRight':
        img = cv2.rotate(img, rot_180)
        

    return img

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

def visualize_puzzle_solver(img,side_type,match_border,match_interior,size=200,plot=True):
    
    """
    For visualizing the solution to the puzzle in a grid-like fasion. 
    
    :param img: A complete list of images for the whole puzzle.
    :param side_type: An array that contains the type configuration of the pieces.
    :param match_border: The list with border matches
    :param match_interior: The list with interior matches
    :param size: For resizing the image so they are square NxN square
    :param plot: Plots the solution using matplotlib
    
    :return: Array with images placed in a grid structure
    
    """
    

    ################################
    ####### Border Assembly ########
    ################################

    # Identify corner pieces
    border = np.where(side_type==0)
    border_unique,counts = np.unique(border[0],return_counts=True)
    corner_pieces = border_unique[counts>1]

    # Resize to square images
    resized_images_org = []
    for i in range(len(img)):
        resized_images_org.append(cv2.resize(img[i],(size,size)))
    
    
    resized_images = resized_images_org.copy()
    
    # For only getting piece index of matched pieces
    match2 = np.zeros((len(match_border),2),dtype=np.int8)
    for i in range(len(match_border)):
        match2[i] = ([match_border[i][0],match_border[i][2]])

    sides = border_frame_assembly(match2,corner_pieces,top_left_corner=1)
    
    top_side = sides[0]
    bottom_side = sides[2]
    bottom_side = bottom_side[::-1]
    left_side = sides[3]
    left_side = left_side[::-1]
    right_side = sides[1]
    
    top_images = []
    bottom_images = []
    for i in range(len(sides[0])-1):
         piece_nr = top_side[i]
         next_image = correct_rotation_border(side_type[piece_nr],resized_images[piece_nr])
         top_images.append(next_image)
         piece_nr = bottom_side[i]
         next_image = correct_rotation_border(side_type[piece_nr],resized_images[piece_nr],rotate='BottomRight')
         bottom_images.append(next_image)

    left_images = []
    right_images = []
    for i in range(len(sides[2])-1):
        piece_nr = left_side[i+1]
        next_image = correct_rotation_border(side_type[piece_nr],resized_images[piece_nr],rotate='BottomLeft')
        left_images.append(next_image)
        piece_nr = right_side[i+1]
        next_image = correct_rotation_border(side_type[piece_nr],resized_images[piece_nr],rotate='TopRight')
        right_images.append(next_image)

    column = len(sides[0])
    rows = len(sides[2])
    # Create an empty canvas
    grid = np.zeros((rows*size,column*size,3),dtype=np.uint8)
    # Places the images in a grid structure
    for i in range(len(top_images)):
        grid[0:size,i*size:(i+1)*size] = top_images[i]
        grid[(rows-1)*size:rows*size,(column-1-i)*size:(column-i)*size] = bottom_images[i]
    for i in range(len(right_images)):
        grid[(rows-1-i)*size:(rows-i)*size,0:size] = left_images[len(left_images)-i-1]
        grid[i*size:(i+1)*size,(column-1)*size:column*size] = right_images[len(left_images)-i-1]
        
    
    ##################################
    ####### Interior Assembly ########
    ##################################

    match_arr = np.array(match_interior)

    # Removes unrelevant matches in top and bottom row
    for i in top_side:
        check1 = np.argwhere(match_arr[:,0]==i)
        check2 = np.argwhere(match_arr[:,2]==i)

        if check1.size>0:
            match_arr = np.delete(match_arr,int(check1),0)
        if check2.size>0:
            match_arr = np.delete(match_arr,int(check2),0)
            
    for i in bottom_side:
        check1 = np.argwhere(match_arr[:,0]==i)
        check2 = np.argwhere(match_arr[:,2]==i)
        if check1.size>0:
            match_arr = np.delete(match_arr,int(check1),0)
        if check2.size>0:
            match_arr = np.delete(match_arr,int(check2),0)
      
    # Finds the sequence for each column in the interior pieces
    interior_sequence = interior_assembly(top_side[1:-1],left_side[1:-1],match_arr)
    left_column_to_matches = left_side[1:-1]
    for i in range(len(interior_sequence)):
        new_column = []
        for j in range(len(interior_sequence[0])):
            piece_idx = interior_sequence[i][j]
            condition1 = np.isin(left_column_to_matches[j], match_arr[:, 0])
            condition2 = np.isin(piece_idx, match_arr[:, 2])
            condition3 = np.isin(left_column_to_matches[j], match_arr[:, 2])
            condition4 = np.isin(piece_idx, match_arr[:, 0])

            # Check if all conditions are true and indices match
            if np.all(condition1 & condition2) and (np.argwhere((match_arr[:, 0] == left_column_to_matches[j]) & (match_arr[:, 2] == piece_idx)).size!=0):
                indices = np.argwhere((match_arr[:, 0] == left_column_to_matches[j]) & (match_arr[:, 2] == piece_idx))
                resized_images[piece_idx] = correct_rotation_interior(resized_images[piece_idx],int(match_arr[(indices[0]), 3]))
                new_column.append(match_arr[int(indices[0]), 2])
            elif np.all(condition3 & condition4) and (np.argwhere((match_arr[:, 2] == left_column_to_matches[j]) & (match_arr[:, 0] == piece_idx)).size!=0):
                indices = np.argwhere((match_arr[:, 2] == left_column_to_matches[j]) & (match_arr[:, 0] == piece_idx))
                resized_images[piece_idx] = correct_rotation_interior(resized_images[piece_idx],int(match_arr[(indices[0]), 1]))
                new_column.append(match_arr[int(indices[0]), 0])
        
        left_column_to_matches = np.array(new_column)
    
    # Place the pieces on the grid
    for i in range(len(interior_sequence)):
        for j in range(interior_sequence[i].size):
            piece_index = interior_sequence[i][j]
            grid[(j+1)*size:(j+2)*size, (i+1)*size:(i+2)*size] = resized_images[piece_index]
           
    if plot:
        plt.figure(1,figsize=(15,15))
        plt.imshow(grid)
        plt.show()
    
    return grid