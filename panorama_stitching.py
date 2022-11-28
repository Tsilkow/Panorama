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
        assert new_camera_matrix is None or (tmp_cmrmtrx == new_camera_matrix).all()
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


def transform_image(in_img, transform):
    out_img = np.zeros_like(in_img)
    inv_trans = np.linalg.inv(transform)

    for x in range(out_img.shape[0]):
        for y in range(out_img.shape[1]):
            origin = inv_trans @ np.array([x, y, 1]).T
            origin = (round(origin[0]), round(origin[1]))
            if origin[0] in range(in_img.shape[0]) and origin[1] in range(in_img.shape[1]):
                out_img[x, y] = in_img[round(origin[0]), round(origin[1])]
            else:
                out_img[x, y] = (0, 0, 0)

    cv2.imwrite(fin_dir+'/transd.png', out_img)


def transform_from_points(point_pairs):
    result = np.zeros((3, 3))
    # A is the final matrix of linear equations for all point pairs
    A = np.empty((0, 9), float)
    for p in point_pairs:
        tmp = (np.array([[         p[0][0],          p[0][1],        1,
                                         0,                0,        0,
                          -p[0][0]*p[1][0], -p[0][1]*p[1][0], -p[1][0]],
                         [               0,                0,        0,
                                   p[0][0],          p[0][1],        1,
                          -p[0][0]*p[1][1], -p[0][1]*p[1][1], -p[1][1]]]))
        A = np.vstack((A, tmp))

    print(A)
    _, _, V = np.linalg.svd(A)
    eingenvector = V[-1, :]
    return eingenvector.reshape(3, 3)


def test_transform_from_points():
    ground_truth = np.random.rand(9)
    ground_truth = (ground_truth / np.linalg.norm(ground_truth, 2)).reshape(3, 3)
    points = [np.random.rand(2) for i in range(6)]
    points = [np.array([p[0], p[1], 1]) for p in points]
    pairs = [(p, ground_truth @ p) for p in points]
    result = transform_from_points(pairs)
    print(ground_truth, result, sep='\n')
    assert (ground_truth == result).all()
    

if __name__ == '__main__':
    # Task 1
    print("Task 1")
    camera_matrix, dist_coeffs = read_coeffs()
    src_filenames = [f for f in os.listdir(src_dir)
                     if f[0] == 'p' and f[-4:] == '.png'][:2]
    assert len(src_filenames) == 2
    camera_matrix, imgs = undistort(src_filenames, camera_matrix, dist_coeffs)

    # cv2.imwrite(fin_dir+'/'+src_filenames[0], imgs[0])
    # cv2.imwrite(fin_dir+'/'+src_filenames[1], imgs[1])

    print('[DONE]')
    # Task 2
    print("Task 2")
    transform = np.array([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
    
    transform_image(imgs[0], transform)

    print('[DONE]')
    # Task 3
    print("Task 3")
    test_transform_from_points()
    print('[DONE]')
