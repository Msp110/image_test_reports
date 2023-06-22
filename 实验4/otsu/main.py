import numpy as np
import cv2 as cv

img = cv.imread('D://yttj.jpg',0)
retVal, a_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
print("使用opencv函数的方法：" + str(retVal))	# 结果为 134
cv.imshow("a_img",a_img)
cv.waitKey()
