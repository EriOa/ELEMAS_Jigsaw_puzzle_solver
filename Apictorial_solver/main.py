import matplotlib.pyplot as plt
from functions import *
from visualization_moomie import *

img, img_gray = Load_images(24,invert=False,binary=True)
side_img,side_type = ReadPieceSides2(img_gray)
hole_interior,head_interior,hole_border,head_border = SortByType(side_type)
match_border = matchPieces(side_img,hole_border,head_border,threshold=10000)
match_interior = matchPieces(side_img,hole_interior,head_interior,threshold=10000)
grid = visualize_puzzle_solver(img,side_type,match_border,match_interior,grid_return=True)
plt.figure(1,figsize=(15,15))
plt.imshow(grid)