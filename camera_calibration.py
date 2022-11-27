import cv2
import numpy as np
import os
import sys

src_dir = 'img_src'
out_file = 'cal_coeffs'

# Scripted designed to calibrate basic distortions of a camera, using a series
# of images (works best with >20) containing a flat laying checker board
# pattern (with 6 by 9 squares) in captured at different perspectives.
#
# Images should be saved in img_src and have the same resolution.
#
# Calibration coefficients are stored in cal_coeffs.
#
# Additionally, undistort function is provided.


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
        img = np.array(cv2.imread(src_dir+'/'+filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("demo", img)
        # cv2.waitKey(100)
        corners = None
        result, corners = cv2.findChessboardCorners(img, pattern_size, corners)
        print(f'{filename}: {result}')
        if result:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS +
                                                 cv2.TERM_CRITERIA_MAX_ITER,
                                                 30, 0.001))
            img_points.append(corners)
            obj_points.append(tmplt_obj)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, img.shape[::-1], None,
                            None)

    return camera_matrix, dist_coeffs


def undistort(filenames, src_dir, out_dir, camera_matrix, dist_coeffs):
    for filename in filenames:
        img = np.array(cv2.imread(src_dir+'/'+filename))
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        print(out_dir+'/'+filename)
        cv2.imwrite(out_dir+'/'+filename, img)


def main():
    cv2.namedWindow("demo")
    calibration_imgs = [f for f in os.listdir(src_dir) if f[-4:] == '.png']
    assert len(calibration_imgs) != 0
    camera_matrix, dist_coeffs = calibrate(calibration_imgs)

    # Uncomment this to undistort images used for calibration
    # undistort(calibration_imgs, src_dir, 'img_cal', camera_matrix, dist_coeffs)

    print(camera_matrix, dist_coeffs, sep='\n')
    og_stdout = sys.stdout
    try:
        f = open(out_file, 'w')
        sys.stdout = f
        print(camera_matrix, dist_coeffs, sep='\n')
    finally:
        sys.stdout = og_stdout


if __name__ == '__main__':
    main()
