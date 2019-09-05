from durgesh.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def edgeDetect(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image = cv2.imread(image)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edged, image, ratio


def contours(edged,image):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image, screenCnt

def perspectiveTransform(orig, screenCnt, ratio):    
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height = 650))
    cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imutils.resize(warped, height = 650)


def segmentation(img,x,y,weidth,height):
    img = img[y:y+height,x:x+weidth]
    cv2.imshow("segmented",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

if __name__ == "__main__":
    
    image_path = "data/accident_forms/MVIMG_20190823_180455.jpg"

    original_image = cv2.imread(image_path)

    edged_image, original_resized_image, ratio= edgeDetect(image_path)
    contoured_image, screencnt= contours(edged_image, original_resized_image)
    perspectiveTransformed_image = perspectiveTransform(original_image, screencnt, ratio)

    x = 170
    y= 81
    weidth = 135
    height = 386

    middle_section = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'Middle_section',(x+2,y+10),0,0.3,(0,255,0))
    

    x = 16
    y = 84
    weidth = 164
    height = 376

    A_section = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'A_section',(x+2,y+10),0,0.3,(0,255,0))

    x = 296
    y = 84
    weidth = 163
    height = 379

    B_section = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'B_section',(x+2,y+10),0,0.3,(0,255,0))

    x = 95
    y = 454
    weidth = 283 
    height = 138

    Drawing = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'Drawing',(x+2,y+10),0,0.3,(0,255,0))

    x = 155
    y = 586
    weidth = 179 
    height = 49

    signature = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'Signature',(x+2,y+10),0,0.3,(0,255,0))
    cv2.line(perspectiveTransformed_image, (x+(weidth//2),y), (x+(weidth//2),y+height), (0,255,0),2)

    x = 18
    y = 30
    weidth = 457 
    height = 58

    top_sction = segmentation(perspectiveTransformed_image,x,y,weidth,height)
    cv2.rectangle(perspectiveTransformed_image,(x,y),(x+weidth,y+height),(0,255,0),2)
    cv2.putText(perspectiveTransformed_image,'Top section',(x+2,y+10),0,0.3,(0,255,0))
    

    cv2.imshow("boxed", perspectiveTransformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
