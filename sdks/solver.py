from pathlib import Path
import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from time import time
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets

from sdks.sdk_cnn import sdk_CNN

show_cv_debug_img = False
test_one_digit = False
show_num_pads = False

class sdkSolver():
    def __init__(self):
        self.sdk_cnn = None
        self.current_img = None
        self.gray_img = None
        self.num_pads = []
        self.board = None
        self.solution = None
        self.output_img = None
        self.mark_points = None

        self.valid = True

        cnn_path = Path('data/sdk_cnn.pt')
        if cnn_path.exists():
            self.sdk_cnn = torch.load(cnn_path)
            self.sdk_cnn.eval()
        else:
            self.train(cnn_path)
            torch.save(self.sdk_cnn, cnn_path)

    def train(self, model_path):
        # load and process dataset
        data_path = Path('data/Chars74K')
        
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
            ])
        
        train_dataset = datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

        # hyperparameters
        epochs = 5
        batch_size = 64
        lr = 0.01
        momentum = 0.9

        network = sdk_CNN()
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)

        # train the model
        start = time()

        for e in range(epochs):
            total_loss = 0
            total_correct = 0

            for batch in train_loader:
                images, labels = batch

                preds = network(images)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                total_correct += get_num_correct(preds, labels)
            else:
                print('Epoch {} - Training loss: {} - Total correct: {} - Accuracy: {}'.format(e, total_loss, total_correct, total_correct / len(train_dataset)))
        end = time()
        delta = end - start
        print('\nTraining Time = ', str(delta // 60), ' m ', str(delta % 60), ' s')

        # save model and model file
        self.sdk_cnn = network
    
    def setup_new_img(self, img_path):
        self.current_img = cv.imread(img_path)

    def get_num_imgs(self):
        # get grayscale image
        gray_img = cv.cvtColor(self.current_img, cv.COLOR_BGR2GRAY)
        gray_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 5)

        # get sudoku mask
        mask = get_mask(gray_img)

        # remove non-sudoku
        clear = cv.bitwise_and(self.current_img, self.current_img, mask=mask)

        # get grid image
        mix_img = get_grid(gray_img, mask)
    
        # find intecept points by contours
        test = np.copy(self.current_img)
        contours, _ = cv.findContours(mix_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mid_points = []
        for contour in contours:
            mid_point = tuple(np.average(contour, axis=0)[0].astype(int))
            mid_points.append(mid_point)
            # draw marks on test image
            cv.circle(test, mid_point, 2, (255, 0, 0), -1)
    
        if show_cv_debug_img:
            cv.imshow('gray', gray_img)
            cv.imshow('clear', clear)
            cv.imshow('mix', mix_img)
            cv.imshow('test', test)

            cv.waitKey(0)
            cv.destroyAllWindows()

        if len(mid_points) != 100:
            print(len(mid_points))
            self.valid = False
            return None

        # sort points
        mid_points.sort(key=lambda a : a[1])
        mid_points = np.array(mid_points).reshape(10, 10, 2).tolist()
        for i in range(10):
            mid_points[i].sort(key=lambda a : a[0])

        # get number pads from points
        self.mark_points = np.copy(mid_points)
        self.num_pads = get_num_pads(gray_img, mid_points)

        return self.num_pads

    def get_sudoku_board_matrix(self):
        if test_one_digit:
            num = self.num_pads[9]
            # num = np.roll(num, 4, axis=0)
            num = clear_num_pad_border(num, 4)
            print(num.shape)
            print(np.sum(num))
            num = cv.resize(num, (128, 128))

            print('score: ',np.sum(num))

            inp = torch.from_numpy(num).float().view(1, 1, 128, 128)

            with torch.no_grad():
                logps = self.sdk_cnn(inp)
            # ps = torch.exp(logps)
            probab = list(logps.numpy()[0])
            pred_label = probab.index(max(probab))
            print(pred_label)

            plt.imshow(num, cmap='gray')
            plt.show()

        # get predict matrix
        pred_mat = []

        # set blank grid threshold
        threshold = 4100000
        for i in range(len(self.num_pads)):
            num = self.num_pads[i]
            num = clear_num_pad_border(num, 4)
            num = cv.resize(num, (128, 128))
            if np.sum(num) < threshold:
                inp = torch.from_numpy(num).float().view(1, 1, 128, 128)

                with torch.no_grad():
                    logps = self.sdk_cnn(inp)
                probab = list(logps.numpy()[0])
                pred_mat.append(probab.index(max(probab)))
            else:
                pred_mat.append(0)
        
        self.board = np.array(pred_mat).reshape(9, 9)

    def solve(self):
        if not valid_board(self.board):
            return None
        answer = np.copy(self.board)
        if solving(answer) == True:
            self.solution = np.copy(answer)
        else:
            self.solution = None

    def mix_board_and_solution(self):
        mix_img = np.copy(self.current_img)

        # set fonts
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 2.3
        color = (255, 0, 0)
        thickness = 2

        # print solution to image
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    mix_img = cv.putText(mix_img, str(self.solution[i][j]),
                                        tuple(self.mark_points[i+1][j]), font,
                                        fontScale, color, thickness, cv.LINE_AA)

        return mix_img

    def run(self):
        self.valid = True
        self.get_num_imgs()
        if not self.valid:
            return None
        if show_num_pads:
            for i in range(9):
                for j in range(9):
                    plt.subplot(9, 9, 9*i+j+1, frameon=False)
                    plt.imshow(self.num_pads[9*i+j], cmap='gray')
                    plt.axis('off')
            plt.show()
        self.get_sudoku_board_matrix()
        self.solve()

        self.valid = self.solution is not None

        if self.valid:
            self.output_img = self.mix_board_and_solution()
        else:
            self.output_img = None

    def get_board(self):
        return self.board

    def get_solution(self):
        return self.solution


# image processing helpers
def get_mask(gray_img):
    # get mask from contours
    mask = np.zeros_like(gray_img)
    contours, _ = cv.findContours(gray_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv.contourArea)[-2]
    cv.drawContours(mask, [largest_contours], 0, 255, -1)

    return mask
def get_grid(gray_img, mask):
    # get grid lines
    sobelx = cv.Sobel(gray_img, cv.CV_8U, 1, 0, ksize=3)
    sobelx = cv.bitwise_and(sobelx, sobelx, mask=mask)
    _, sobelx = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY)
    sobely = cv.Sobel(gray_img, cv.CV_8U, 0, 1, ksize=3)
    sobely = cv.bitwise_and(sobely, sobely, mask=mask)
    _, sobely = cv.threshold(sobely, 0, 255, cv.THRESH_BINARY)

    kernelx = cv.getStructuringElement(cv.MORPH_RECT,(2,15))
    kernely = cv.getStructuringElement(cv.MORPH_RECT,(15,2))
    sobelx = cv.morphologyEx(sobelx, cv.MORPH_DILATE, kernelx)
    sobely = cv.morphologyEx(sobely, cv.MORPH_DILATE, kernely)

    linesx_img = sobelx.copy()
    contour, hier = cv.findContours(linesx_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv.boundingRect(cnt)
        if h/w > 15:
            cv.drawContours(linesx_img,[cnt],0,255,-1)
        else:
            cv.drawContours(linesx_img,[cnt],0,0,-1)
    sobelx = cv.morphologyEx(linesx_img,cv.MORPH_DILATE,None,iterations = 2)

    linesy_img = sobely.copy()
    contour, hier = cv.findContours(linesy_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv.boundingRect(cnt)
        if w/h > 15:
            cv.drawContours(linesy_img,[cnt],0,255,-1)
        else:
            cv.drawContours(linesy_img,[cnt],0,0,-1)
    sobely = cv.morphologyEx(linesy_img,cv.MORPH_DILATE,None,iterations = 2)

    mix_img = cv.bitwise_and(linesx_img, linesy_img)
    kernel = np.ones((5, 5), np.uint8)
    mix_img = cv.dilate(mix_img, kernel, iterations=3)

    if show_cv_debug_img:
        cv.imshow('sobelx', sobelx)
        cv.imshow('sobely', sobely)
        cv.imshow('img_x', linesx_img)
        cv.imshow('img_y', linesy_img)

    return mix_img
def get_num_pads(gray_img, mid_points):
    num_pads = []

    for i in range(9):
        for j in range(9):
            num_border = np.array([[mid_points[i][j],
                                    mid_points[i][j+1],
                                    mid_points[i+1][j+1],
                                    mid_points[i+1][j]]],
                                    dtype='int32')

            num_border = num_border[0]
            x0 = max(num_border[0][0], num_border[3][0])
            x1 = min(num_border[1][0], num_border[2][0])
            y0 = max(num_border[0][1], num_border[1][1])
            y1 = min(num_border[2][1], num_border[3][1])
            num_pad = np.array(gray_img[y0:y1, x0:x1])
            num_pads.append(num_pad)
    
    return num_pads
def clear_num_pad_border(pad, size):
    result = np.copy(pad)
    result[:size, :] = 255
    result[-size:, :] = 255
    result[:, :size] = 255
    result[:, -size:] = 255
    return result
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

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
def solving(board):
    # board is full
    pos = find_empty(board)
    if pos is None:
        return True

    for num in range(1, 10):
        board[pos[0]][pos[1]] = num
        if check_valid(board, pos) == True:
            if solving(board) == True:
                return True

        board[pos[0]][pos[1]] = 0

    return False

if __name__ == "__main__":
    solver = sdkSolver()
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

    answer = solver.solve(board)
    if answer is not None:
        print((answer==correct_answer).all())
