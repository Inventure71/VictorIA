




def find_possible_position_of_next_move(matrix):
    # array with height of each possible move for each column
    # ES:
    # matrix = [0,0,0,0,0,0,0] means the board is empty
    # matrix = [1,0,0,0,0,0,0] means there is a circle in the first column and the next will go at height 1
    rows_n = 6
    cols_n = 7

    columns_height = [-1 for _ in range(cols_n)]

    for row_index in range((rows_n-1), -1, -1):
        for col_index in range(cols_n):
            if columns_height[col_index] == -1:
                if matrix[row_index][col_index] == 0:
                    columns_height[col_index] = row_index

    print(columns_height)

    return columns_height
