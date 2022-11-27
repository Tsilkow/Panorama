import cv2
import keyboard
from camera import RESOLUTIONS, Camera
import os


# Simple script aimed at capturing images using small cameras given out during
# labs
#
# Before running make sure that camera is powered and this comuputer is
# connected to the camera's wifi -- if nothing happens when you run the script,
# this might be the issue
#
# Press s to save image
# Press + and - to adjust resolution (its value will be printed in the
# terminal)

def main():
    cv2.namedWindow("demo")
    cam = Camera()
    img_counter = 0
    os.chdir("img/")

    print("Connected, starting continuous capture")
    while True:
        cam.keep_stream_alive()
        img = cam.get_frame()
        cv2.imshow("demo", img)
        cv2.waitKey(1)
        if keyboard.is_pressed("q"):
            break
        elif keyboard.is_pressed("s"):
            cv2.imwrite(f'{img_counter:03d}.png', img)
            print(f'Image was saved to img/{img_counter:03d}.png')
            img_counter += 1
        elif keyboard.is_pressed("+"):
            q = cam.get_quality()
            if not q == max(RESOLUTIONS.keys()):
                cam.set_quality(q + 1)
                print(f'Changed quality to {RESOLUTIONS[q+1]}')
        elif keyboard.is_pressed("-"):
            q = cam.get_quality()
            if not q == min(RESOLUTIONS.keys()):
                cam.set_quality(q - 1)
                print(f'Changed quality to {RESOLUTIONS[q-1]}')


if __name__ == '__main__':
    main()
