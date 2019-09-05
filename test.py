import cv2
import imutils 

image = cv2.imread("EAS-Formular-1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = imutils.resize(gray, height = 500)
cv2.imshow("greyed", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
