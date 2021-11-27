import cv2
import numpy as np
from stereo_camera.errors import ChessboardNotFoundError
from concurrent.futures import ProcessPoolExecutor
import typing


def process_stereo_image(
    img_name_pair: typing.Tuple[str, str], pattern_size: typing.Tuple[int, int]
) -> (str, np.ndarray, np.ndarray):
    """Processes a right/left pair of images and detects chessboard corners for calibration
    :param img_name_pair: tuple containing the names of the right and left images, respectively
    :param pattern_size: tuple containing the number of internal corners of the chessboard

    :returns the name of the stereo image
    :returns the tuple containing two NumPy arrays, each representing the corners
    of the right and left images
    :returns the tuple of stereo images of a chessboard with corners drawn"""
    stereo_img_name = img_name_pair[0].split("/")[-1:][0]
    print(f"Processing image {stereo_img_name}...")

    # Process in parallel both images
    with ProcessPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(process_image_task, img_name_pair[0], pattern_size)
        futureR = executor.submit(process_image_task, img_name_pair[1], pattern_size)
        # Collect the images with the detected chessboard and the corners, which will be used for calibration
        img_pointsL, img_drawn_cornersL = futureL.result()
        img_pointsR, img_drawn_cornersR = futureR.result()
    stereo_img_points = (img_pointsL, img_pointsR)
    stereo_img_drawn_corners = (img_drawn_cornersL, img_drawn_cornersR)

    print(f"Image {stereo_img_name} processed")
    return stereo_img_name, stereo_img_points, stereo_img_drawn_corners


def process_image_task(
    img_name: str, pattern_size: typing.Tuple[int, int]
) -> (np.ndarray, np.ndarray):
    # Find chessboard corners
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # If the chessboard is not found, an exception is raised
    if not found:
        raise ChessboardNotFoundError(*img_name.split("/")[-1:])

    # Refine corners position
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    img_drawn_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img_drawn_corners, pattern_size, corners, found)

    return corners.reshape(-1, 2), img_drawn_corners


def compute_disparity(
    dstL: np.ndarray,
    dstR: np.ndarray,
    stereo_matcher: cv2.StereoSGBM,
    disp_bounds: typing.Optional[typing.List[float]] = None,
) -> np.ndarray:
    """Given a pair of stereo images, already undistorted and rectified, this function computes the disparity map
    :param dstL: NumPy array representing the left image (already undistorted and rectified)
    :param dstR: NumPy array representing the right image (already undistorted and rectified)
    :param stereo_matcher: SemiGlobal Block Matching stereo matcher based on the left image
    :param disp_bounds: list containing global minimum and maximum of disparity map,
                        useful to avoid "jumping color" problem (optional)

    :return disp: grayscale disparity map"""

    # Compute disparity and normalize w.r.t. global max and min
    disp = stereo_matcher.compute(dstL, dstR)
    if disp_bounds is not None:
        disp_bounds[0] = min(disp_bounds[0], disp.min())
        disp_bounds[1] = max(disp_bounds[1], disp.max())
        disp = (disp - disp_bounds[0]) * (65535.0 / (disp_bounds[1] - disp_bounds[0]))
        disp = cv2.convertScaleAbs(disp, alpha=(255.0 / 65535.0))
    else:
        disp = cv2.normalize(
            src=disp,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    # Apply bilateral filter to disparity map
    disp = cv2.bilateralFilter(disp, d=5, sigmaColor=100, sigmaSpace=100)
    # Refine with close operator
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
    return disp
