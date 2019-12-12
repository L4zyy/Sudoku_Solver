
import numpy as np

show_cv_debug_img = False
test_one_digit = False
show_num_pads = False

# sudoku matrix solving helpers
def valid_cell(cell):
    l = cell.reshape(-1).tolist()
    return not any(l.count(x) > 1 and x != 0 for x in l)
def valid_board(board):
    for i in range(9):
        if not valid_cell(board[i]):
            return False

    for i in range(9):
        if not valid_cell(board[:, i]):
            return False

    for i in range(3):
        for j in range(3):
            if not valid_cell(board[i*3:i*3+3, j*3: j*3+3]):
                return False
    return True
def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None
def check_valid(board, pos):
    # axis 0
    for i in range(9):
        if i != pos[0] and board[i][pos[1]] == board[pos[0]][pos[1]]:
            return False
    # axis 1
    for j in range(9):
        if j != pos[1] and board[pos[0]][j] == board[pos[0]][pos[1]]:
            return False
    # grid
    grid_i = pos[0] // 3
    grid_j = pos[1] // 3

    for i in range(grid_i*3, grid_i*3+3):
        for j in range(grid_j*3, grid_j*3+3):
            if (i, j) != pos and board[i][j] == board[pos[0]][pos[1]]:
                return False

    return True
def solve(board):
    # return Validity, modify board to solution

    # board is full
    pos = find_empty(board)
    if pos is None:
        return True

    for num in range(1, 10):
        board[pos[0]][pos[1]] = num
        if check_valid(board, pos) == True:
            if solve(board) == True:
                return True

        board[pos[0]][pos[1]] = 0

    return False

if __name__ == "__main__":
    board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ])
    correct_answer = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ])

    answer = solve(board)
    print(answer)
    print(board)
    if answer is not None:
        print(np.array_equal(board, correct_answer))
