import cv2
import numpy as np
import os

dir = 'img'
cal_prefix = 'c_'
val_prefix = 'v_'

def checker_points(width: int, size: tuple):
    assert len(size) == 2
    result = np.zeros((size[0] * size[1], 3), np.float32)
    result[:, :2] = width * np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    return result

def calibrate(filenames):
    obj_points = []
    img_points = []
    pattern_size = (5, 8)
    tmplt_obj = checker_points(30, pattern_size)
    for filename in filenames:
        img = np.array(cv2.imread(filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("demo", img)
        #cv2.waitKey(100)
        corners = None
        result, corners = cv2.findChessboardCorners(img, pattern_size, corners)
        print(f'{filename}: {result}')
        if result:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)
            obj_points.append(tmplt_obj)
            cv2.drawChessboardCorners(img, pattern_size, corners, True)
            #cv2.imwrite(f'out_{filename}', img)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None, None)

    return camera_matrix, dist_coeffs

def undistort(filenames, camera_matrix, dist_coeffs):
    for filename in filenames:
        img = np.array(cv2.imread(filename))
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        print(f'out_{filename}')
        cv2.imwrite(f'out_{filename}', img)

def main():
    cv2.namedWindow("demo")
    calibration_imgs = [f'{dir}/{f}' for f in os.listdir(dir) if cal_prefix in f]
    validation_imgs = [f'{dir}/{f}' for f in os.listdir(dir) if val_prefix in f]
    assert len(calibration_imgs) != 0
    assert len(validation_imgs) != 0
    camera_matrix, dist_coeffs = calibrate(calibration_imgs)
    print(f'Camera Matrix = \n{camera_matrix}')
    print(f'Distortion Coefficients = \n{dist_coeffs}')
    undistort(validation_imgs, camera_matrix, dist_coeffs)

if __name__ == '__main__':
    main()
