# Panorama

This project is for an assignment for Robot Control course.
Goal is to capture an two images using a provided camera, undistort them (using calibration data acquired in a previous excersize, however the code needed to do so is included here) and stitch them together to create a panorama.

## How to run
0. If you want to use new images, you have to provide distortions coefficients in `cal_coeff.json`. You can use `camera_calibration.py` script for that.
1. Place undistorted images in `img_src` under names `p1.png` and `p2.png`
2. Run `panorama_stitcher.py`
3. View in `img_final`:
   * undistorted original images under `p1.png` and `p2.png`
   * Stitched image from task 5 under name `task_5_panorama.png`
   * Stitched image from task 7 under name `task_7_panorama.png`