import cv2
import numpy as np
import os
import json
from json import JSONEncoder


src_dir = 'img_src'
fin_dir = 'img_final'
coeff_file = 'cal_coeffs.json'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def undistort(filenames, camera_matrix, dist_coeffs):
    result_imgs = []
    new_camera_matrix = None
    for filename in filenames:
        img = np.array(cv2.imread(src_dir+'/'+filename))
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        h, w = img.shape[:2]
        tmp_cmrmtrx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        assert new_camera_matrix is None or tmp_cmrmtrx == new_camera_matrix
        new_camera_matrix = tmp_cmrmtrx
        x, y, w, h = roi
        result_imgs.append(img[y:y+h, x:x+w])
    return new_camera_matrix, result_imgs


def read_coeffs():
    camera_matrix, dist_coeffs = None, None
    with open(coeff_file, 'r') as f:
        np_data = json.load(f)
        camera_matrix = np.asarray(np_data['camera_matrix'])
        dist_coeffs = np.asarray(np_data['distortion_coefficients'])

    assert camera_matrix is not None
    assert dist_coeffs is not None

    return camera_matrix, dist_coeffs


if __name__ == '__main__':
    camera_matrix, dist_coeffs = read_coeffs()
    src_filenames = [f for f in os.listdir(src_dir)
                     if f[0] == 'p' and f[-4:] == '.png'][:2]
    assert len(src_filenames) == 2
    imgs = undistort(src_filenames, camera_matrix, dist_coeffs)
