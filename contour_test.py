import cv2
import numpy


img=numpy.zeros((200, 200), dtype=numpy.uint8)
img[50:150, 50:150]=255

cv2.imshow("img", img)

ret, thresh=cv2.threshold(img, 127, 255, 0)
cv2.imshow("thresh", thresh)
image, contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

color=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imshow("color", color)

img=cv2.drawContours(color, contours, -1, (0, 255, 0), 2)

cv2.imshow("1111111", img)
cv2.imshow("2222222", color)

cv2.waitKey(-1)
cv2.destroyAllWindows()
