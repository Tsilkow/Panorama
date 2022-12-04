import cv2
import numpy as np
import os
import json
import math
import random
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
    inv_trans = normalize_homography(np.linalg.inv(transform))
    in_corners = np.array([[0, 0], [0, in_img.shape[1]],
                           list(in_img.shape[:2]), [in_img.shape[0], 0]]).T
    in_corners = np.vstack((in_corners, np.ones((1, 4))))
    out_corners = np.around(normalize_points(transform @ in_corners)).astype(int)
    out_offset = np.array([int(round(min(out_corners[0, :]))),
                           int(round(min(out_corners[1, :]))),
                           0])
    out_img = np.zeros((-out_offset[0] + int(round(max(out_corners[0, :]))),
                        -out_offset[1] + int(round(max(out_corners[1, :]))),
                        3))
    print(f'offset = {out_offset}')
    print(f'out_corners = {out_corners}')
    print(f'in_img.shape = {in_img.shape}')
    print(f'out_img.shape = {out_img.shape}')

    coords = np.array(np.meshgrid(range(out_offset[0], out_img.shape[0] + out_offset[0]),
                                  range(out_offset[1], out_img.shape[1] + out_offset[1]),
                                  range(1, 2)))
    coords = coords.reshape(coords.shape[:3]).T
    origin = np.around(normalize_points((inv_trans @ coords.reshape(-1, 3).T)).T.reshape(out_img.shape)).astype(int)
    domain = ((origin[:, :, 0] >= 0) & (origin[:, :, 0] <  in_img.shape[0]) & \
              (origin[:, :, 1] >= 0) & (origin[:, :, 1] <  in_img.shape[1]))
    origin[np.logical_not(domain)] = 0
    out_img[:, :, 0] = np.where(domain, in_img[origin[:, :, 0], origin[:, :, 1], 0], 0)
    out_img[:, :, 1] = np.where(domain, in_img[origin[:, :, 0], origin[:, :, 1], 1], 0)
    out_img[:, :, 2] = np.where(domain, in_img[origin[:, :, 0], origin[:, :, 1], 2], 0)
 
    cv2.imwrite(fin_dir+'/transd.png', out_img)
    return out_img, out_offset


def normalize_point(point):
    if len(point) == 2: return (point[0], point[1], 1)
    elif len(point) == 3: return point / point[2]
    else: raise ValueError
    

def normalize_points(points):
    norm = points[2, :]
    assert (norm != 0).all
    result = points / norm
    return result


def normalize_homography(homography):
    homography = homography.reshape(-1)
    assert homography.shape[0] == 9
    return (homography / np.linalg.norm(homography, 2)).reshape(3, 3)


def transform_from_points(point_pairs):
    result = np.zeros((3, 3))
    # A is the final matrix of linear equations for all point pairs
    A = np.empty((0, 9), float)
    for pair in point_pairs:
        if len(pair[0]) == 2:
            x, y = pair[0]
        elif len(pair[0]) == 3:
            x, y, _ = pair[0]
        else: raise ValueError
        
        if len(pair[1]) == 2:
            u, v = pair[1]
        elif len(pair[1]) == 3:
            u, v, _ = pair[1]
        else: raise ValueError
        
        tmp = np.array([[x, y, 1, 0, 0, 0, -x*u, -y*u, -u],
                        [0, 0, 0, x, y, 1, -x*v, -y*v, -v]])
        A = np.vstack((A, tmp))

    _, _, V = np.linalg.svd(A)
    eingenvector = normalize_homography(V[-1, :])
    return eingenvector


def test_transform_from_points(threshold):
    ground_truth = normalize_homography(np.random.rand(9))
    test_points = [np.random.rand(2) for i in range(4)]
    test_points = [np.array([p[0], p[1], 1]) for p in test_points]
    test_pairs = [(p, normalize_point(ground_truth @ p)) for p in test_points]
    inferred = transform_from_points(test_pairs)
    validation_points = [np.random.rand(2) for i in range(20)]
    validation_points = [np.array([p[0], p[1], 1]) for p in validation_points]
    comparison = [(normalize_point(ground_truth @ p),
                   normalize_point(inferred @ p))
                  for p in validation_points]
    difference = [c[0] - c[1] for c in comparison]
    error = np.linalg.norm(difference, 1)
    assert error < threshold


def stitch_images(in_imgs, matching_points):
    transform = transform_from_points(matching_points)
    trans_imgs = [in_imgs[0], in_imgs[1]]
    trans_imgs[0], offset = transform_image(in_imgs[0], transform)
    #offset = np.array([offset[1], offset[0], offset[2]])
    #offset = [normalize_point(transform @ normalize_point(pair[0])) - normalize_point(pair[1]) for pair in matching_points]
    #offset = np.around(np.mean(offset, axis=0)).astype(int)[:2]
    print(offset)
    i1x = [max(0,  offset[0]), max(0,  offset[0]) + trans_imgs[0].shape[0]]
    i1y = [max(0,  offset[1]), max(0,  offset[1]) + trans_imgs[0].shape[1]]
    i2x = [max(0, -offset[0]), max(0, -offset[0]) + trans_imgs[1].shape[0]]
    i2y = [max(0, -offset[1]), max(0, -offset[1]) + trans_imgs[1].shape[1]]
    print(i1x, i1y, i2x, i2y, sep='\n')
    corners = np.array([[[i1x[0], i1y[0]], [i1x[1], i1y[0]],
                         [i1x[0], i1y[1]], [i1x[1], i1y[1]]],
                        [[i2x[0], i2y[0]], [i2x[1], i2y[0]],
                         [i2x[0], i2y[1]], [i2x[1], i2y[1]]]])
    print(corners)
    width  = int(round(max(i1x[1], i2x[1]) - min(i1x[0], i2x[0])))
    height = int(round(max(i1y[1], i2y[1]) - min(i1y[0], i2y[0])))
    out_img = np.zeros((width, height, 3))
    print(f'offset = {offset}')
    print(f'out_img.shape = {out_img.shape}')
    print(f'trans_imgs[0].shape = {trans_imgs[0].shape}')
    print(f'trans_imgs[1].shape = {trans_imgs[1].shape}')
    out_img[i1x[0]:i1x[1], i1y[0]:i1y[1]] += trans_imgs[0]//2
    out_img[i2x[0]:i2x[1], i2y[0]:i2y[1]] += trans_imgs[1]//2
    cv2.imwrite(fin_dir+'/'+'final.png', out_img)    
    

def get_matches(img1, img2, visualize=True, lowe_ratio=0.6):
    # Convert imgaes to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("vis", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches



def get_best_matches(matching_points, threshold, tries=1000):
    best_inliners = []
    for t in range(tries):
        inliners = []
        sample = random.sample(matching_points, 4)
        transform = transform_from_points(sample)
        for pair in matching_points:
            error = np.linalg.norm(normalize_point(pair[1]) -
                                   normalize_point(transform @ (pair[0][0], pair[0][1], 1)), 1)
            if error < threshold:
                inliners.append(pair)
        if len(best_inliners) < len(inliners): best_inliners = inliners
    return best_inliners
    

def task_1():
    print("Task 1")
    camera_matrix, dist_coeffs = read_coeffs()
    src_filenames = [f for f in os.listdir(src_dir)
                     if f[0] == 'p' and f[-4:] == '.png'][:2]
    assert len(src_filenames) == 2
    camera_matrix, imgs = undistort(src_filenames, camera_matrix, dist_coeffs)
    cv2.imwrite(fin_dir+'/'+src_filenames[0], imgs[0])
    cv2.imwrite(fin_dir+'/'+src_filenames[1], imgs[1])
    print('[DONE]')
    return imgs


def task_2(imgs):
    print("Task 2")
    transform = normalize_homography(np.array([[ 1   , -0.5 , -0.2 ],
                                               [-0.5 ,  1   , -0.2 ],
                                               [ 0.01,  0.01,  1   ]]))
    print(transform)
    transform_image(imgs[0], transform)
    print('[DONE]')


def task_3():
    print("Task 3")
    for i in range(1000):
        test_transform_from_points(10**(-6))
    print("[DONE]")


def task_4():
    print("Task 4")
    matching_points = [((322, 445), (587, 457)),
                       ((352, 447), (591, 456)),
                       (( 30, 488), (319, 472)),
                       ((245, 110), (512, 121)),
                       ((180,  55), (453,  72))]
    # matching_points = [((215, 124), (483, 136)),
    #                    ((315, 475), (577, 488)),
    #                    (( 63, 534), (347, 518)),
    #                    ((266, 110), (535, 120)),
    #                    ((399,  79), (679,  81))]
    transform = transform_from_points(matching_points)
    print(transform)
    print("[DONE]")
    return transform


def task_5(imgs, transform):
    print("Task 5")
    stitch_images(imgs, transform)
    print("[DONE]")
    

def task_6(imgs):
    print("Task 6")
    matches = get_matches(imgs[0], imgs[1], False)
    print("[DONE]")
    return matches


def task_7(imgs, matching_points):
    print("Task 7")
    best_points = get_best_matches(matching_points, 0.1)
    print(len(best_points))
    stitch_images(imgs, best_points)
    print("[DONE]")


if __name__ == '__main__':
    imgs = task_1()
    #task_2(imgs)
    task_3()
    #transform = task_4()
    #task_5(imgs, transform)
    matching_points = task_6(imgs)
    print(len(matching_points))
    task_7(imgs, matching_points)
