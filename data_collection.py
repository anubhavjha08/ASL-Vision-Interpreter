import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)  # Initialize webcam capture

detector = HandDetector(maxHands=1)  # Detect up to 1 hand at a time

offset = 20  # Padding for the cropped hand image
img_size = 300  # Size for the blank canvas (white image)

while True:
    success, img = cap.read()  # Capture frame from webcam

    hands, img = detector.findHands(img)  # Detect hands and return updated frame

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping region stays within image boundaries
        if y - offset >= 0 and x - offset >= 0 and y + h + offset <= img.shape[0] and x + w + offset <= img.shape[1]:
            img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop hand region

            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255  # Create a blank white canvas

            aspect_ratio = h / w

            # Resize the cropped image based on the aspect ratio
            if aspect_ratio > 1:  # Tall image (height > width)
                k = img_size / h
                width_calculated = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (width_calculated, img_size))
                img_resize_shape = img_resize.shape
                width_gap = math.ceil((img_size-width_calculated)/2)
                img_white[0: img_resize_shape[0], width_gap: width_calculated+width_gap] = img_resize  # Paste resized image
            else:
                k = img_size / w
                height_calculated = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (img_size, height_calculated))
                img_resize_shape = img_resize.shape
                height_gap = math.ceil((img_size-height_calculated)/2)
                img_white[height_gap: height_calculated+height_gap, 0: img_resize_shape[1]] = img_resize  # Paste resized image


            # Display the cropped hand image and the resized one on the white canvas
            cv2.imshow("ImageCrop", img_crop)
            cv2.imshow("ImageCrop-White", img_white)

    cv2.imshow("Image", img)  # Display the original frame
    cv2.waitKey(1)  # Process next frame after 1ms
