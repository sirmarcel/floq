import numpy as np

# Provide functions to get/set blocks in numpy arrays

def get_block_from_matrix(matrix,dim_block,n_block,row,column):
    start_row = row*dim_block
    start_column = column*dim_block

    stop_row = start_row+dim_block
    stop_column = start_column+dim_block

    return matrix[start_row:stop_row,start_column:stop_column]

def set_block_in_matrix(block,matrix,dim_block,n_block,row,column):
    start_row = row*dim_block
    start_column = column*dim_block

    stop_row = start_row+dim_block
    stop_column = start_column+dim_block

    matrix[start_row:stop_row,start_column:stop_column] = block