# Code by Billal Tech
# Copyright (c) 2020 Billal Fauzan

import cv2
import numpy as np
import sys
import time

def main():
    try:
        path = sys.argv[1]
    except IndexError:
        print ("[ERROR]: Masukan file gambar!")
        sys.exit()

    image = cv2.imread(path)

    cascade_fullbody = cv2.CascadeClassifier("/home/billal/Project/python/OpenCV/media/haarcascade_fullbody.xml")
    cascade_eye = cv2.CascadeClassifier("/home/billal/Project/python/OpenCV/media/haarcascade_eye.xml")
    cascade_face = cv2.CascadeClassifier("/home/billal/Project/python/OpenCV/media/haarcascade_frontalface_default.xml")

    while True:
        width = int(image.shape[1] * 60 / 100)
        height = int(image.shape[0] * 60 / 100)

        image = cv2.resize(image, (width, height))
        body = cascade_fullbody.detectMultiScale(
            image,
            minSize=(20, 20)
        )

        face = cascade_face.detectMultiScale(
            image,
            minSize=(20, 20)
        )

        for (x, y, w, h) in body:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0), 2)
            cv2.putText(image, "Badan", (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)

        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0), 2)
            cv2.putText(image, "Wajah", (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        
        print ("Width: " + str(width))
        print ("Height: " + str(height))
        time.sleep(1)

        cv2.putText(image, "Copyright (c) Billal Tech", (width - 520, height - 750), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,200), 1)
        cv2.imshow("Result", image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()