import cv2
import numpy as np

def harris_corner_detection(image_path, threshold=0.01):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    
    image[dst > threshold * dst.max()] = [0, 0, 255]

    for y, x in np.argwhere(dst > threshold * dst.max()):
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  

    cv2.imshow('Harris Corner Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

harris_corner_detection('building.jpg')
