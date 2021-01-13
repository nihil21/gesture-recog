import cv2
import numpy as np
from model.errors import ChessboardNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional


def process_stereo_image(img_name_pair: Tuple[str, str],
                         pattern_size: Tuple[int, int]) -> (str, np.ndarray, np.ndarray):
    """Processes a right/left pair of images and detects chessboard corners for calibration
        :param img_name_pair: tuple containing the names of the right and left images, respectively
        :param pattern_size: tuple containing the number of internal corners of the chessboard

        :returns the name of the stereo image
        :returns the tuple containing two NumPy arrays, each representing the corners
        of the right and left images
        :returns the tuple of stereo images of a chessboard with corners drawn"""
    stereo_img_name = img_name_pair[0].split('/')[-1:][0]
    print('Processing image {0:s}...'.format(stereo_img_name))

    # Process concurrently both images
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(process_image_thread, img_name_pair[0], pattern_size)
        futureR = executor.submit(process_image_thread, img_name_pair[1], pattern_size)
        # Collect the images with the detected chessboard and the corners, which will be used for calibration
        for future in as_completed([futureL, futureR]):
            if future == futureL:
                img_pointsL, img_drawn_cornersL = futureL.result()
            else:
                img_pointsR, img_drawn_cornersR = futureR.result()
        stereo_img_points = (img_pointsL, img_pointsR)
        stereo_img_drawn_corners = (img_drawn_cornersL, img_drawn_cornersR)

    print('Image {0:s} processed'.format(stereo_img_name))
    return stereo_img_name, stereo_img_points, stereo_img_drawn_corners


def process_image_thread(img_name: str,
                         pattern_size: Tuple[int, int]) -> (np.ndarray, np.ndarray):
    # Find chessboard corners
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # If the chessboard is not found, an exception is raised
    if not found:
        raise ChessboardNotFoundError(*img_name.split('/')[-1:])

    # Refine corners position
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    img_drawn_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img_drawn_corners, pattern_size, corners, found)

    return corners.reshape(-1, 2), img_drawn_corners


def compute_disparity(frameL: np.ndarray,
                      frameR: np.ndarray,
                      mapxL: np.ndarray,
                      mapxR: np.ndarray,
                      mapyL: np.ndarray,
                      mapyR: np.ndarray,
                      stereo_matcherL: cv2.StereoSGBM,
                      stereo_matcherR: cv2.StereoMatcher,
                      wls_filter: cv2.ximgproc_DisparityWLSFilter,
                      valid_ROI: Optional[Tuple[int, int, int, int]] = None) -> (np.ndarray,
                                                                                 np.ndarray,
                                                                                 np.ndarray,
                                                                                 np.ndarray):
    """This function computes the undistorted and rectified left and right frames of a stereo camera,
    and the disparity maps, both in grayscale and colored
        :param frameL: NumPy array representing the left image
        :param frameR: NumPy array representing the right image
        :param mapxL: NumPy array representing the homography to undistort and rectify the left image along x-axis
        :param mapxR: NumPy array representing the homography to undistort and rectify the right image along x-axis
        :param mapyL: NumPy array representing the homography to undistort and rectify the left image along y-axis
        :param mapyR: NumPy array representing the homography to undistort and rectify the right image along y-axis
        :param stereo_matcherL: SemiGlobal Block Matching stereo matcher based on the left image
        :param stereo_matcherR: stereo matcher based on the right image, useful to filter the disparity map
        :param wls_filter: Weighted Least Squares filter, useful to improve the quality of the disparity map
        :param valid_ROI: tuple of four integers representing the Region Of Interest
                          common to both left and right images (optional)

        :return disp: colored disparity map
        :return disp_gray: grayscale disparity map
        :return dstL: NumPy array representing the undistorted and rectified left image
        :return dstR: NumPy array representing the undistorted and rectified right image"""
    # Undistort and rectify
    dstL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    dstR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    # Crop frames if valid_ROI is provided
    if valid_ROI is not None:
        x, y, w, h = valid_ROI
        dstL = dstL[y:y + h, x:x + w]
        dstR = dstR[y:y + h, x:x + w]

    # Compute disparities concurrently using the grayscale version with enhanced contrast
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(stereo_matcherL.compute, dstL, dstR)
        futureR = executor.submit(stereo_matcherR.compute, dstR, dstL)
        futures = {futureL, futureR}
        for future in as_completed(futures):
            if future == futureL:
                dispL = future.result()
            else:
                dispR = future.result()
    filtered_disp = wls_filter.filter(dispL, dstL, None, dispR)
    disp_gray = cv2.normalize(src=filtered_disp,
                              dst=None,
                              alpha=0,
                              beta=255,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
    # Apply colormap to disparity
    disp = cv2.applyColorMap(disp_gray, cv2.COLORMAP_JET)
    return disp, disp_gray, dstL, dstR


def compute_depth(disp_point: int,
                  baseline: float,
                  alpha_uL: float,
                  alpha_uR: float,
                  u_0L: float,
                  u_0R: float) -> float:
    """This function, given a disparity value of a point and the camera parameters,
    computes the corresponding depth of such point
    :param disp_point: integer representing the disparity value of a point
    :param baseline: float representing the distance between the two cameras
    :param alpha_uL: float representing the alpha_u intrinsic parameter of the left camera
    :param alpha_uR: float representing the alpha_u intrinsic parameter of the right camera
    :param u_0L: float representing the u_0 intrinsic parameter of the left camera
    :param u_0R: float representing the u_0 intrinsic parameter of the right camera

    :return depth: float representing the depth of the given point"""

    # Compute the average alpha_u
    alpha_u = (alpha_uL + alpha_uR) / 2
    # Compute the depth
    return alpha_u * baseline / (disp_point + u_0R - u_0L)


def process_segment():
    seg_img = cv2.imread('../disp-samples/Hand.jpg')
    gray_seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray_seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(seg_img, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Img", seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
