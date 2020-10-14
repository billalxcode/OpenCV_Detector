#Code by Billal Tech
#~copyright (c) 2020 Billal Fauzan~

import cv2
import sys
import numpy as np

def main():
    try:
        path = sys.argv[1]
    except IndexError:
        print ("[ERROR]: Masukan file gambar...")
        sys.exit()

    cascade = cv2.CascadeClassifier("media/cars.xml")
    new_frame = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        
    while True:

        print ("Original shape: " + str(new_frame.shape))
        gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        detect = cascade.detectMultiScale(
            gray,
            minSize=(20, 20)
        )

        for (x, y, w, h) in detect:
            cv2.rectangle(new_frame, (x, y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Hello", new_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

if __name__ == "__main__":
    main()
