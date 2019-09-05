from __future__ import print_function
import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local



MAX_MATCHES = 80000
GOOD_MATCH_PERCENT = 0.005


def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  cv2.imshow("matches", imMatches)
  cv2.waitKey(0)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h

def makeitwhite(img):
  #-----Converting image to LAB Color model----------------------------------- 
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  cv2.imshow("lab",lab)
  cv2.waitKey(0)
  #-----Splitting the LAB image to different channels-------------------------
  l, a, b = cv2.split(lab)
  cv2.imshow('l_channel', l)
  cv2.imshow('a_channel', a)
  cv2.imshow('b_channel', b)
  cv2.waitKey(0)

  #-----Applying CLAHE to L-channel-------------------------------------------
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  cl = clahe.apply(l)
  cv2.imshow('CLAHE output', cl)
  cv2.waitKey(0)
  #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
  limg = cv2.merge((cl,a,b))
  cv2.imshow('limg', limg)
  cv2.waitKey(0)
  #-----Converting image from LAB Color model to RGB model--------------------
  final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  cv2.imshow('final', final)
  cv2.waitKey(0)

  return final



if __name__ == '__main__':
  
  # Read reference image
  refFilename = r"scanned_final_2.png"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  imReference = imutils.resize(imReference, height= 1500)
  # Read image to be aligned
  imFilename = r"data/accident_forms/MVIMG_20190823_180455.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  im = imutils.resize(im, height= 1500)
  
  cv2.imshow("reference", imReference)
  cv2.waitKey(0)
  cv2.imshow("to be alligned", im)
  cv2.waitKey(0)
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
  
  # Write aligned image to disk. 
  outFilename = "aligned_2.jpg"
  print("Saving aligned image : ", outFilename); 

  imReg = makeitwhite(imReg)

  imReg = cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY)
  T = threshold_local(imReg, 11, offset = 10, method = "gaussian")
  imReg = (imReg > T).astype("uint8") * 255


  cv2.imwrite(outFilename, imReg)
  cv2.imshow("alligned", imReg)
  cv2.waitKey(0)

  # Print estimated homography
  print("Estimated homography : \n",  h)

cv2.waitKey(0)
cv2.destroyAllWindows()
  
  