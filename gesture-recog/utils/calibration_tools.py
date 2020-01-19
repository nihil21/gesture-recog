import cv2
import numpy as np
import zmq
import base64
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt
from exceptions.error import ChessboardNotFoundError


def capture_images(socks: Dict[str, zmq.Socket],
                   folders: Dict[str, str]) -> None:
    """Function which coordinates the capture of images from both cameras
        :param socks: dictionary containing the two zmq sockets for the two clients, identified by a label
        :param folders: dictionary containing the folder in which images will be saved, identified by a label"""
    print('Collecting images of a chessboard for calibration...')

    # Receive confirmation by the client
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['DX'].recv_string)
        executor.submit(socks['SX'].recv_string)

    # Receive frames by both clients using threads
    print("Both cameras are ready")
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(capture_images_thread, socks['DX'], folders['DX'], 'DX')
        executor.submit(capture_images_thread, socks['SX'], folders['SX'], 'SX')
    print('Images collected')


# noinspection PyUnresolvedReferences
def capture_images_thread(sock: zmq.Socket,
                          folder: str,
                          sock_idx: str) -> None:
    """Captures a series of images from the remote camera
        :param sock: zmq socket to communicate with the remote camera
        :param folder: path of the folder in which images will be saved
        :param sock_idx: index of the socket ('DX' or 'SX')"""
    # Send signal to synchronize both cameras
    sock.send_string("Start signal received")

    # Initialize variables for countdown
    n_pics, tot_pics = 0, 30
    n_sec, tot_sec = 0, 4
    str_sec = '4321'
    start_time = datetime.now()

    # Loop until 'tot_pics' images are collected
    while n_pics < tot_pics:
        # Read frame as a base64 string
        serial_frame = sock.recv_string()
        buffer = base64.b64decode(serial_frame)
        frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)

        # Display counter on screen before saving frame
        if n_sec < tot_sec:
            # Draw on screen the current remaining seconds
            frame = cv2.putText(img=frame,
                                text=str_sec[n_sec],
                                org=(int(40), int(80)),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=3,
                                color=(255, 255, 255),
                                thickness=5,
                                lineType=cv2.LINE_AA)

            # If time elapsed is greater than one second, update 'n_sec'
            time_elapsed = (datetime.now() - start_time).total_seconds()
            if time_elapsed >= 1:
                n_sec += 1
                start_time = datetime.now()
        else:
            # When countdown ends, save grayscale image to file
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            path = folder
            cv2.imwrite(path + '{:02d}'.format(n_pics) + '.jpg', gray_frame)
            n_pics += 1
            n_sec = 0
            print('{}: {:d}/{:d} images collected'.format(sock_idx, n_pics, tot_pics))

        cv2.imshow('{} frame'.format(sock_idx), frame)
        cv2.waitKey(1)  # invocation of non-blocking 'waitKey', required by OpenCV after 'imshow'
        if n_pics == tot_pics:
            # If enough images are collected, the termination signal is sent to the client
            sock.send_string('\0')

    # Try to read the last frames, if any
    while True:
        try:
            sock.recv_string(flags=zmq.NOBLOCK)
            break
        except zmq.Again:
            pass

    cv2.destroyAllWindows()


# noinspection PyUnresolvedReferences
def calibrate(folders: Dict[str, str],
              pattern_size: Tuple[int, int],
              square_length: float,
              calib_folders: Dict[str, str]) -> None:
    """Calibrates both cameras using pictures of a chessboard
        :param folders: dictionary containing the folder in which images will be saved, identified by a label
        :param pattern_size: size of the chessboard used for calibration
        :param square_length: float representing the length, in mm, of the square edge
        :param calib_folders: dictionary containing the file path in which calibration data will be saved,
                              identified by a label"""

    # Get a list of images captured, divided by folder
    dx_img_names, sx_img_names = glob.glob(folders['DX'] + '*.jpg'), glob.glob(folders['SX'] + '*.jpg')
    # Produce a list of pairs '(right_image, left_image)'
    # If one list has more elements than the other, the extra elements will be automatically discarded by 'zip'
    img_pair_names = list(zip(dx_img_names, sx_img_names))
    # Process concurrently stereo images
    corners_pairs = []
    img_corners_pairs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_stereo_image,
                                   img_pair_name,
                                   pattern_size)
                   for img_pair_name in img_pair_names]

        for future in as_completed(futures):
            try:
                corners_pair, img_corners_pair = future.result()
                corners_pairs.append(corners_pair)
                img_corners_pairs.append(img_corners_pair)
            except ChessboardNotFoundError as e:
                print('No chessboard found in image {}'.format(e.image))

    # Produce two lists of corners, one for the right and one for the left cameras
    corners_pairs_unzipped = [list(t) for t in zip(*corners_pairs)]
    dx_corners = corners_pairs_unzipped[0]
    sx_corners = corners_pairs_unzipped[1]

    # Get image shape
    h, w = img_corners_pairs[0][0].shape[:2]

    # Calibrate concurrently both cameras and get the distortion mapping
    print('Calibrating both cameras...')
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dx = executor.submit(calibrate_single_camera,
                                    dx_corners,
                                    pattern_size,
                                    square_length,
                                    (w, h),
                                    calib_folders['DX'])
        future_sx = executor.submit(calibrate_single_camera,
                                    sx_corners,
                                    pattern_size,
                                    square_length,
                                    (w, h),
                                    calib_folders['SX'])
        dx_mapx, dx_mapy, dx_roi = future_dx.result()
        sx_mapx, sx_mapy, sx_roi = future_sx.result()

    # Ask user to plot corrected test images
    while True:
        disp_images = input('Do you want to plot the images used for calibration? [y/n]: ')
        if disp_images not in ['y', 'n']:
            print('The option inserted is not valid, retry.')
        else:
            break

    # Plot the images with corners drawn on the chessboard and with calibration applied
    if disp_images == 'y':
        for img_corners_pair in img_corners_pairs:
            fig, ax = plt.subplots(nrows=2, ncols=2)
            # Plot the original images
            ax[0][0].imshow(img_corners_pair[1])
            ax[0][0].set_title('SX frame')
            ax[0][1].imshow(img_corners_pair[0])
            ax[0][1].set_title('DX frame')

            # Apply mapping to images
            dx_dst = cv2.remap(img_corners_pair[0], dx_mapx, dx_mapy, cv2.INTER_LINEAR)
            sx_dst = cv2.remap(img_corners_pair[1], sx_mapx, sx_mapy, cv2.INTER_LINEAR)

            # Crop only the region of interest
            x, y, w, h = dx_roi
            dx_dst = dx_dst[y:y + h, x:x + w]
            x, y, w, h = sx_roi
            sx_dst = sx_dst[y:y + h, x:x + w]

            # Plot the undistorted images
            ax[1][0].imshow(sx_dst)
            ax[1][0].set_title('SX frame undistorted')
            ax[1][1].imshow(dx_dst)
            ax[1][1].set_title('DX frame undistorted')
            plt.show()
    print('Calibration done')


def process_stereo_image(img_pair_name: Tuple[str, str],
                         pattern_size: Tuple[int, int]) -> (np.ndarray, np.ndarray):
    """Processes a right/left pair of images and detects chessboard corners for calibration
        :param img_pair_name: tuple containing the names of the right and left images, respectively
        :param pattern_size: tuple containing the number of internal corners of the chessboard

        :returns corners_pair: tuple containing two NumPy arrays, each representing the corners
                               of the right and left images"""
    stereo_img_name = img_pair_name[0].split('/')[-1:]
    print('Processing image {}'.format(stereo_img_name))

    # Process in parallel both images
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dx = executor.submit(process_image_thread, img_pair_name[0], pattern_size)
        future_sx = executor.submit(process_image_thread, img_pair_name[1], pattern_size)
        # Collect the images with the detected chessboard and the corners, which will be used for calibration
        dx_corners, dx_img_corners = future_dx.result()
        sx_corners, sx_img_corners = future_sx.result()
        corners_pair = (dx_corners, sx_corners)
        img_corners_pair = (dx_img_corners, sx_img_corners)

    print('Image {} processed'.format(stereo_img_name))
    return corners_pair, img_corners_pair


# noinspection PyUnresolvedReferences
def process_image_thread(img_name: str,
                         pattern_size: Tuple[int, int]) -> (np.ndarray, np.ndarray):
    # Find chessboard corners
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # If the chessboard is not found, an exception is raised
    if not found:
        raise ChessboardNotFoundError(img_name.split('/')[-1:])

    # Refine corners position
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    img_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img_corners, pattern_size, corners, found)

    return corners.reshape(-1, 2), img_corners


# noinspection PyUnresolvedReferences
def calibrate_single_camera(corners: List[np.ndarray],
                            pattern_size: Tuple[int, int],
                            square_length: float,
                            camera_size: Tuple[int, int],
                            calib_folder: str) -> (float, float, np.ndarray):
    """Maps coordinates of corners in 2D image to corners in 3D image and calibrates a camera
        :param corners: list of the chessboard corners detected in the images used for calibration
        :param pattern_size: size of the chessboard used, expressed in term of internal corners
        :param square_length: length in mm of the edge of a square of the chessboard
        :param camera_size: size, in pixel, of images
        :param calib_folder: file path in which calibration data will be saved"""

    # Prepare object points by obtaining a grid, scaling it by the square edge length and reshaping it
    pattern_points = np.zeros([pattern_size[0] * pattern_size[1], 3], dtype=np.float32)
    pattern_points[:, :2] = (np.indices(pattern_size, dtype=np.float32) * square_length).T.reshape(-1, 2)

    # Create lists for 2D and 3D points
    img_points = []  # 2D points in image plane
    obj_points = []  # 3D points in real world space
    for corner_points in corners:
        img_points.append(corner_points)
        obj_points.append(pattern_points)

    # Obtain camera parameters, such as camera matrix and rotation/translation vectors
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                                       img_points,
                                                                       camera_size,
                                                                       None,
                                                                       None)
    # Compute new optimal camera matrix
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, camera_size, 1, camera_size)
    # Compute the mapping between undistorted and distorted images
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, new_camera_mtx, camera_size, 5)

    # Save all camera parameters to .npy files
    print('Saving calibration data to file...')
    np.save(file=calib_folder + 'rms',
            arr=rms,
            allow_pickle=False)
    np.save(file=calib_folder + 'camera_matrix',
            arr=camera_matrix,
            allow_pickle=False)
    np.save(file=calib_folder + 'dist_coefs',
            arr=dist_coefs,
            allow_pickle=False)
    np.save(file=calib_folder + 'rvecs',
            arr=rvecs,
            allow_pickle=False)
    np.save(file=calib_folder + 'tvecs',
            arr=tvecs,
            allow_pickle=False)
    np.save(file=calib_folder + 'new_camera_mtx',
            arr=new_camera_mtx,
            allow_pickle=False)
    np.save(file=calib_folder + 'roi',
            arr=roi,
            allow_pickle=False)
    np.save(file=calib_folder + 'mapx',
            arr=mapx,
            allow_pickle=False)
    np.save(file=calib_folder + 'mapy',
            arr=mapy,
            allow_pickle=False)

    # Returns mapping
    return mapx, mapy, roi


# TODO
def disp_map():
    pass
