import cv2
import numpy as np
import zmq
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt
from exceptions.error import ChessboardNotFoundError
from utils.network_tools import concurrent_send, recv_frame


def capture_images(socks: Dict[str, zmq.Socket],
                   folders: Dict[str, str]) -> None:
    """Function which coordinates the capture of images from both cameras
        :param socks: dictionary containing the two zmq sockets for the two slaves, identified by a label
        :param folders: dictionary containing the folder in which images will be saved, identified by a label"""
    print('Collecting images of a chessboard for calibration...')

    # Receive confirmation by the slave
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['L'].recv_string)
        executor.submit(socks['R'].recv_string)

    # Receive frames by both slaves using threads
    print("Both cameras are ready")
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(capture_images_thread, socks['L'], folders['L'], 'L')
        executor.submit(capture_images_thread, socks['R'], folders['R'], 'R')
    print('Images collected')


# noinspection PyUnresolvedReferences
def capture_images_thread(sock: zmq.Socket,
                          folder: str,
                          camera_idx: str) -> None:
    """Captures a series of images from the remote camera
        :param sock: zmq socket to communicate with the remote camera
        :param folder: path of the folder in which images will be saved
        :param camera_idx: index of the socket ('L'/'R')"""
    # Send signal to synchronize both cameras
    sock.send_string("Start signal received")

    # Initialize variables for countdown
    n_pics, tot_pics = 0, 30
    n_sec, tot_sec = 0, 4
    str_sec = '4321'
    start_time = datetime.now()

    # Loop until 'tot_pics' images are collected
    while n_pics < tot_pics:
        frame = recv_frame(sock)

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
            print('{}: {:d}/{:d} images collected'.format(camera_idx, n_pics, tot_pics))

        cv2.imshow('{} frame'.format(camera_idx), frame)
        cv2.waitKey(1)  # invocation of non-blocking 'waitKey', required by OpenCV after 'imshow'
        if n_pics == tot_pics:
            # If enough images are collected, the termination signal is sent to the slave
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
              calib_file: str) -> None:
    """Calibrates both cameras using pictures of a chessboard
        :param folders: dictionary containing the folder in which images will be saved, identified by a label
        :param pattern_size: size of the chessboard used for calibration
        :param square_length: float representing the length, in mm, of the square edge
        :param calib_file: path to the file in which calibration data will be saved"""

    # Get a list of images captured, divided by folder
    img_namesL, img_namesR = glob.glob(folders['L'] + '*.jpg'), glob.glob(folders['R'] + '*.jpg')
    # Produce a list of pairs '(right_image, left_image)'
    # If one list has more elements than the other, the extra elements will be automatically discarded by 'zip'
    img_name_pairs = list(zip(img_namesL, img_namesR))
    # Process concurrently stereo images
    stereo_img_names = []
    stereo_img_points_list = []
    stereo_img_drawn_corners_list = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_stereo_image,
                                   img_name_pair,
                                   pattern_size)
                   for img_name_pair in img_name_pairs]

        for future in as_completed(futures):
            try:
                stereo_img_name, stereo_img_points, stereo_img_drawn_corners = future.result()
                stereo_img_names.append(stereo_img_name)
                stereo_img_points_list.append(stereo_img_points)
                stereo_img_drawn_corners_list.append(stereo_img_drawn_corners)
            except ChessboardNotFoundError as e:
                print('No chessboard found in image {}'.format(e.image))

    # Produce two lists of image points, one for the right and one for the left cameras
    stereo_img_points_unzipped = [list(t) for t in zip(*stereo_img_points_list)]
    img_pointsL = stereo_img_points_unzipped[0]
    img_pointsR = stereo_img_points_unzipped[1]

    # Prepare object points by obtaining a grid, scaling it by the square edge length and reshaping it
    pattern_points = np.zeros([pattern_size[0] * pattern_size[1], 3], dtype=np.float32)
    pattern_points[:, :2] = (np.indices(pattern_size, dtype=np.float32) * square_length).T.reshape(-1, 2)
    # Append them in a list with the same length as the image points' one
    obj_points = []
    for i in range(0, len(img_pointsL)):
        obj_points.append(pattern_points)

    # Get camera size
    h, w = stereo_img_drawn_corners_list[0][0].shape[:2]

    # Calibrate concurrently single cameras and get the camera intrinsic parameters
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(calibrate_single_camera,
                                  img_pointsL,
                                  obj_points,
                                  (w, h),
                                  'L')
        futureR = executor.submit(calibrate_single_camera,
                                  img_pointsR,
                                  obj_points,
                                  (w, h),
                                  'R')
        cam_mtxL, distL = futureL.result()
        cam_mtxR, distR = futureR.result()

    # Use intrinsic parameters to calibrate more reliably the stereo camera
    print('Calibrate stereo camera...')
    flag = cv2.CALIB_FIX_INTRINSIC
    error, cam_mtxL, distL, cam_mtxR, distR, rot_mtx, trasl_mtx, e_mtx, f_mtx = cv2.stereoCalibrate(obj_points,
                                                                                                    img_pointsL,
                                                                                                    img_pointsR,
                                                                                                    cam_mtxL,
                                                                                                    distL,
                                                                                                    cam_mtxR,
                                                                                                    distR,
                                                                                                    (w, h),
                                                                                                    flags=flag)
    print('Stereo camera calibrated, error: {}'.format(error))
    rot_mtxL, rot_mtxR, proj_mtxL, proj_mtxR, disp_to_depth_mtx, valid_ROIL, valid_ROIR = cv2.stereoRectify(cam_mtxL,
                                                                                                            distL,
                                                                                                            cam_mtxR,
                                                                                                            distR,
                                                                                                            (w, h),
                                                                                                            rot_mtx,
                                                                                                            trasl_mtx)
    # Compute the undistorted and rectify mapping
    mapxL, mapyL = cv2.initUndistortRectifyMap(cam_mtxL, distL, rot_mtxL, proj_mtxL, (w, h), cv2.CV_32FC1)
    mapxR, mapyR = cv2.initUndistortRectifyMap(cam_mtxR, distR, rot_mtxR, proj_mtxR, (w, h), cv2.CV_32FC1)

    # Save all camera parameters to .npy files
    np.savez_compressed(file=calib_file, cam_mtxL=cam_mtxL, cam_mtxR=cam_mtxR, disp_to_depth_mtx=disp_to_depth_mtx,
                        distL=distL, distR=distR, mapxL=mapxL, mapxR=mapxR, mapyL=mapyL, mapyR=mapyR,
                        proj_mtxL=proj_mtxL, proj_mtxR=proj_mtxR, rot_mtx=rot_mtx, rot_mtxL=rot_mtxL, rot_mtxR=rot_mtxR,
                        trasl_mtx=trasl_mtx, valid_ROIL=valid_ROIL, valid_ROIR=valid_ROIR)

    print('Calibration data saved to file')

    # Ask user to plot the images used for calibration with the applied corrections
    while True:
        disp_images = input('Do you want to plot the images used for calibration? [y/n]: ')
        if disp_images not in ['y', 'n']:
            print('The option inserted is not valid, retry.')
        else:
            break

    # Plot the images with corners drawn on the chessboard and with calibration applied
    if disp_images == 'y':
        for i in range(0, len(stereo_img_names)):
            # for stereo_img_drawn_corners in stereo_img_drawn_corners_list:
            stereo_img_drawn_corners = stereo_img_drawn_corners_list[i]
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.suptitle(stereo_img_names[i])
            # Plot the original images
            ax[0][0].imshow(stereo_img_drawn_corners[0])
            ax[0][0].set_title('L frame')
            ax[0][1].imshow(stereo_img_drawn_corners[1])
            ax[0][1].set_title('R frame')

            # Remap images using the mapping found after calibration
            dstL = cv2.remap(stereo_img_drawn_corners[0], mapxL, mapyL, cv2.INTER_LINEAR)
            dstR = cv2.remap(stereo_img_drawn_corners[1], mapxR, mapyR, cv2.INTER_LINEAR)

            # Crop ROI
            #valid_ROI = cv2.getValidDisparityROI(valid_ROIL, valid_ROIR, 4, 128, 15)
            #x, y, w, h = valid_ROI
            #dstL = dstL[y:y + h, x:x + w]
            #dstR = dstR[y:y + h, x:x + w]

            # Plot the undistorted images
            ax[1][0].imshow(dstL)
            ax[1][0].set_title('L frame undistorted')
            ax[1][1].imshow(dstR)
            ax[1][1].set_title('R frame undistorted')

            plt.show()


def process_stereo_image(img_name_pair: Tuple[str, str],
                         pattern_size: Tuple[int, int]) -> (str, np.ndarray, np.ndarray):
    """Processes a right/left pair of images and detects chessboard corners for calibration
        :param img_name_pair: tuple containing the names of the right and left images, respectively
        :param pattern_size: tuple containing the number of internal corners of the chessboard

        :returns stereo_img_points: tuple containing two NumPy arrays, each representing the corners
                                    of the right and left images
        :returns stereo_img_drawn_corners: tuple of stereo images of a chessboard with corners drawn"""
    stereo_img_name = img_name_pair[0].split('/')[-1:]
    print('Processing image {}'.format(stereo_img_name))

    # Process in parallel both images
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(process_image_thread, img_name_pair[0], pattern_size)
        futureR = executor.submit(process_image_thread, img_name_pair[1], pattern_size)
        # Collect the images with the detected chessboard and the corners, which will be used for calibration
        img_pointsL, img_drawn_cornersL = futureL.result()
        img_pointsR, img_drawn_cornersR = futureR.result()
        stereo_img_points = (img_pointsL, img_pointsR)
        stereo_img_drawn_corners = (img_drawn_cornersL, img_drawn_cornersR)

    print('Image {} processed'.format(stereo_img_name))
    return stereo_img_name, stereo_img_points, stereo_img_drawn_corners


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
    img_drawn_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img_drawn_corners, pattern_size, corners, found)

    return corners.reshape(-1, 2), img_drawn_corners


# noinspection PyUnresolvedReferences
def calibrate_single_camera(img_points: List[np.ndarray],
                            obj_points: List[np.ndarray],
                            camera_size: Tuple[int, int],
                            camera_idx: str) -> (np.ndarray, np.ndarray):
    """Maps coordinates of corners in 2D image to corners in 3D image and calibrates a camera
        :param img_points: list of the image points
        :param obj_points: list of the object points
        :param camera_size: size, in pixel, of images
        :param camera_idx: string representing the label of the camera

        :return cam_mtx: camera matrix computed by calibration process
        :return dist: distortion coefficient of the camera"""
    print('Calibrating {} camera...'.format(camera_idx))

    # Obtain camera parameters, such as camera matrix and rotation/translation vectors
    rms, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                           img_points,
                                                           camera_size,
                                                           None,
                                                           None)
    print('{}: camera calibrated, RMS = {}'.format(camera_idx, str(rms)))

    # Returns camera matrix and distortion coefficient
    return cam_mtx, dist


# TODO
# noinspection PyUnresolvedReferences
def disp_map(socks: Dict[str, zmq.Socket],
             calib_file: str):
    # Load calibration data
    calib_data = np.load(calib_file + '.npz')
    mapxL = calib_data['mapxL']
    mapyL = calib_data['mapyL']
    mapxR = calib_data['mapxR']
    mapyR = calib_data['mapyR']
    valid_ROIL = calib_data['valid_ROIL']
    valid_ROIR = calib_data['valid_ROIR']

    # Create stereo matcher
    range_disp = 4
    min_disp = 0
    num_disp = 16 * (range_disp - min_disp)
    window_size = 9

    # Create and configure stereo matcher
    stereo_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)  # 64, 9
    stereo_matcher.setPreFilterCap(63)
    stereo_matcher.setPreFilterSize(15)
    stereo_matcher.setROI1(tuple(valid_ROIL))
    stereo_matcher.setROI2(tuple(valid_ROIR))
    stereo_matcher.setTextureThreshold(500)

    # Receive confirmation by cameras
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['L'].recv_string)
        executor.submit(socks['R'].recv_string)
    print('Both cameras are ready')
    # Synchronize cameras
    concurrent_send(socks, msg='Start')

    # Get frames from both cameras
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(recv_frame, socks['L'])
        futureR = executor.submit(recv_frame, socks['R'])
        frameL = cv2.cvtColor(futureL.result(), cv2.COLOR_BGR2GRAY)
        frameR = cv2.cvtColor(futureR.result(), cv2.COLOR_BGR2GRAY)

    # Undistort and rectify
    dstL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    dstR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    # Compute disparity and valid ROI
    disparity = stereo_matcher.compute(dstL, dstR)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    #cv2.imwrite('L.jpg', dstL)
    #cv2.imwrite('R.jpg', dstR)
    #cv2.imwrite('disp.jpg', disp)
    #cv2.imwrite('disp_color.jpg', disp_color)

    cv2.imshow('L frame', dstL)
    cv2.imshow('R frame', dstR)
    cv2.imshow('Disparity', disp)
    cv2.imshow('Disparity Color', disp_color)
    cv2.waitKey(0)

    '''''
    TODO: real-time disparity map
    # Receive confirmation by cameras
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['L'].recv_string)
        executor.submit(socks['R'].recv_string)
    print('Both cameras are ready')
    # Synchronize cameras
    concurrent_send(socks, msg='Start')

    while True:
        # Get frames from both cameras
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(recv_frame, socks['L'])
            futureR = executor.submit(recv_frame, socks['R'])
            frameL = futureL.result()
            frameR = futureR.result()
            
        # Undistort and rectify
        dstL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
        dstR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

        # Compute disparity and valid ROI
        disparity = stereo_matcher.compute(dstL, dstR)
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(cv2.imshow, 'L frame', dstL)
            executor.submit(cv2.imshow, 'R frame', dstR)
        cv2.imshow('Disparity map', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            concurrent_send(socks, '\0')
            break

    # Try to read the last frames, if any
    while True:
        try:
            socks['L'].recv_string(flags=zmq.NOBLOCK)
            socks['R'].recv_string(flags=zmq.NOBLOCK)
            break
        except zmq.Again:
            pass
    '''''

    cv2.destroyAllWindows()
