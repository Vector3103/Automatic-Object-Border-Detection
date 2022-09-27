from ctypes.wintypes import RGB
import cv2
import numpy as np
import keyboard
from rembg.bg import remove

from PIL import Image, ImageDraw, ImageFilter

img = cv2.imread("C:\\Users\\ishu3\\Downloads\\1.jpg",1)



r = cv2.selectROI("select the area", img)
print(r)


cropped_image = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# cv2.imshow("Cropped image", cropped_image)
# cv2.waitKey(0)
result = remove(cropped_image)

cv2.imshow("resul", result)
cv2.waitKey(0)
img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# cv2.imshow("resul", img_gray)
# cv2.waitKey(0)
# ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

out = np.zeros_like(img_gray)

cv2.drawContours(cropped_image, contours, -1,(255,0,0), 3)


def combine_two_color_images_with_anchor(image1, image2, anchor_y, anchor_x):
    foreground, background = image1.copy(), image2.copy()
    #check
    background_height = background.shape[0]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background boundaries at this location")
    
    alpha =0

    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width
    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    cv2.imshow('composited image', background)

    cv2.waitKey(0)

imgf = combine_two_color_images_with_anchor(cropped_image, img, 1,1)

cv2.imshow('Result1', imgf)
cv2.waitKey(c)
if keyboard.read_key() == "c":
    cv2.imshow("original", img)
    cv2.waitKey(0)

print(img.shape)
print(cropped_image.shape)