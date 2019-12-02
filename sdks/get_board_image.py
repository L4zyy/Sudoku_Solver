import numpy as np
import matplotlib.pyplot as plt
import cv2

def thresholding(image):
    if len(image.shape) > 2:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        result = np.copy(image)
    ret, result = cv2.threshold(result, 170, 255, cv2.THRESH_BINARY)
    # result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)

    return result

image = cv2.imread('data/image_1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)
mask = cv2.imread('data/mask_1.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((20, 20), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=5)

mask_3c = np.dstack((mask, mask, mask))
mix = cv2.addWeighted(image, 0.4, mask_3c, 0.6, 0.)

kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(canny, kernel, iterations=1)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)

target_img = np.zeros_like(edges)
target_img[mask!=0] = edges[mask!=0]
target_img = thresholding(target_img)
target_img[mask==0] = 0

cnts, _ = cv2.findContours(target_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
best_approx = None
max_area = 0
area_threshold = 100
if len(cnts) == 0:
    print("No contours were found.")
else:
    for c in cnts:
        area = cv2.contourArea(c)
        if area > area_threshold:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
            if area > max_area and len(approx) == 4:
                best_approx = approx
                max_area = area


detection_img = np.copy(image)
cv2.drawContours(detection_img, [best_approx], 0, (0, 255, 0), 3)
for point in best_approx:
    x = point[0][0]
    y = point[0][1]
    cv2.circle(detection_img, (x, y), 3, (0, 0, 255), -1)

def rectify(approx):
    approx = approx.reshape((4, 2))
    approx_new = np.zeros((4, 2), dtype=np.float32)

    add = approx.sum(1)
    approx_new[0] = approx[np.argmin(add)] # Top Left
    approx_new[2] = approx[np.argmax(add)] # Bottom Right

    diff = np.diff(approx, axis=1)
    approx_new[1] = approx[np.argmin(diff)] # Top Right
    approx_new[3] = approx[np.argmax(diff)] # Bottom Left

    return approx_new

def separate_num_pads(img, step, border_width=5):
    new_step = step + border_width
    new_size = new_step * 9
    result = np.ones((new_size, new_size, 3), np.uint8) * 255

    for i in range(9):
        for j in range(9):
            result[i*new_step : (i+1)*new_step-border_width, j*new_step : (j+1)*new_step-border_width] = \
            img[i*step : (i+1)*step, j*step : (j+1)*step]

    return result

def get_num_pads(img, step):
    pads = []

    for i in range(9):
        for j in range(9):
            pad = np.copy(img[i*step : (i+1)*step, j*step : (j+1)*step])
            pads.append(pad)

    return pads

best_approx = rectify(best_approx)
pad_size = 64
rect_size = 9 * pad_size
target_rect = np.array([[0, 0], [rect_size-1, 0], [rect_size-1, rect_size-1], [0, rect_size-1]], np.float32)
retval = cv2.getPerspectiveTransform(best_approx, target_rect)
warp = cv2.warpPerspective(image, retval, (rect_size, rect_size))
sep = separate_num_pads(warp, pad_size)

pads = get_num_pads(warp, pad_size)

for pad in pads:
    plt.imshow(pad)
    plt.show()

height, width, _ = image.shape
height = int(height/3)
width = int(width/3)

image_show = cv2.resize(image, (width, height))
mask_show = cv2.resize(mask_3c, (width, height))
mix_show = cv2.resize(mix, (width, height))
# canny_show = cv2.resize(canny, (width, height))
canny_show = canny

cv2.imshow('image', image_show)
cv2.imshow('mask', mask_show)
cv2.imshow('mix', mix_show)
cv2.imshow('target', target_img)
cv2.imshow('detection', detection_img)
cv2.imshow('warp', warp)
cv2.imshow('sep', sep)
cv2.waitKey(0)
cv2.destroyAllWindows()