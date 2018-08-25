import numpy as np
import cv2
     
# Load an color image in grayscale
img = cv2.imread('aguia.jpg')

cv2.imshow('image',img)

print len(img[0])
print len(img)
print img[0]
print img


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()