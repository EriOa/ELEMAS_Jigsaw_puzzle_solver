from categorization import *
from functions import *
from visualizing_yoshi import *
from scoring_functions_colour import *


bin,col = load_img(49)
side_img,pieceSides_col, sideType = readPieceSides_col(col,bin)
hole_interior,head_interior,hole_border,head_border = SortByType(sideType)
match_interior = colour_mathing_LAP(side_img,pieceSides_col,hole_interior,head_interior)
match_border = colour_mathing_LAP(side_img,pieceSides_col,hole_border,head_border)
grid = visualize_puzzle_solver(col,sideType,match_border,match_interior,plot=True)



