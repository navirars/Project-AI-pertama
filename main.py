import numpy as np
import cv2
import glob

img_dir1 = "Dataset/NANGKA/"
img_dir2 = "Dataset/SIRIH/"

ext = ['jpg']

files1 = []
files2 = []

[files1.extend(glob.glob(img_dir1 + '*.' + e)) for e in ext]
[files2.extend(glob.glob(img_dir2 + '*.' + e)) for e in ext]

images1 = [cv2.imread(file1) for file1 in files1]
images2 = [cv2.imread(file2) for file2 in files2]

#adjust contrast to all of them
x = 1
for img1 in images1:
    img_adjusted1 = cv2.addWeighted(img1, 1.5, np.zeros(img1.shape, img1.dtype), 0, -1)
    img_name1 = "Dataset/nangka-contrast/" + str(x) + ".jpg"
    cv2.imwrite(img_name1, img_adjusted1)
    x += 1

y = 1
for img2 in images2:
    img_adjusted2 = cv2.addWeighted(img2, 1.5, np.zeros(img2.shape, img2.dtype), 0, 5)
    img_name2 = "Dataset/sirih-contrast/" + str(y) + ".jpg"
    cv2.imwrite(img_name2, img_adjusted2)
    y += 1