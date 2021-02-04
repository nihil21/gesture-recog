from model.network_agent import ImageReceiver
from utils.image_proc_tools import compute_disparity, process_stereo_image
from model.errors import *
import cv2
import numpy as np
import os
import glob
import psutil
import time
import matplotlib.pyplot as plt
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Tuple, Optional


class StereoCamera:
    """Class representing a StereoCamera, composed by two ImageReceiver objects; it enables and simplifies the
    concurrent communication with both sensors.

    Attributes:
        _left_sensor --- ImageSender objects related to the left sensor
        _right_sensor --- ImageSender objects related to the right sensor
        _stereo_frames --- attribute in which the most recent frames are stored
        _data_ready --- event that notifies the main process if a new pair of frames has been read
        _kill_thread --- event that kills the IO thread
        _io_thread --- IO thread that concurrently reads from the two sensor's SUB sockets and saves the frames
                       to the _stereo_frames field
        _calib_params --- dictionary containing the calibration parameters (initially empty)
        _disp_params --- dictionary containing the disparity parameters (initially empty)
        _is_calibrated --- boolean variable indicating whether the stereo camera is calibrated or not
        (initially set to 'False')
        _has_disparity_params --- boolean variable indicating whether the stereo camera has disparity
        _parameters set or not (initially set to 'False')
        _disp_bounds --- tuple of floats representing the minimum and maximum disparity values detected

    Methods:
        multicast_send_sig --- method which enables the concurrent sending of a control signal via the
        two ImageReceiver objects
        multicast_recv_sig --- method which enables the concurrent reception of a control signal via the
        two ImageReceiver objects
        _create_io_thread --- method which creates and starts an IO thread to read the frames in background
        _kill_io_thread --- method which kills the IO thread
        _read_stereo_frames_in_background --- target of the IO thread: it repeatedly reads new frames from both sensors
        _recv_stereo_frames --- method which returns the most recent frames
        load_calib_params --- method which reads the calibration parameters from a given file
        _set_calib_params --- method which sets/updates the calibration parameters
        _save_calib_params --- method which persists the calibration parameters to a given file
        load_disp_params --- method which reads the disparity parameters from a given file
        _set_disp_params --- method which sets/updates the disparity parameters
        _save_disp_params --- method which persists the disparity parameters to a given file
        _flush_pending_frames --- method which flushes both image sockets
        close --- method which releases the network resources used by the two ImageReceiver objects
        capture_sample_images --- method which captures sample images for calibration
        calibrate --- given the sample images, it computes the calibration parameters"""

    def __init__(self,
                 ip_addrL: str,
                 ip_addrR: str,
                 img_port: int,
                 ctrl_port: int):
        # Set up one ImageReceiver object for each sensor
        self._left_sensor = ImageReceiver(ip_addrL, img_port, ctrl_port)
        self._right_sensor = ImageReceiver(ip_addrR, img_port, ctrl_port)
        # Set up an IO thread to read from the two sensors
        self._io_thread = None
        self._data_ready_event = Event()
        self._kill_thread_event = Event()
        self._stereo_frames: Optional[Tuple[float, np.ndarray]] = None
        # Calibration data
        self._calib_params = {}
        self._disp_params = {}
        self._is_calibrated = False
        self._has_disparity_params = False
        # Keep track of min and max disparities
        self._disp_bounds = [np.inf, -np.inf]

    def multicast_send_sig(self, sig: bytes):
        """Method which enables the concurrent sending of a signal to both ImageReceiver objects
            :param sig: string representing the signal to be sent concurrently to both sensors"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self._left_sensor.send_sig, sig)
            executor.submit(self._right_sensor.send_sig, sig)

    def multicast_recv_sig(self) -> (str, str):
        """Method which enables the concurrent reception of a signal from both ImageSender objects
            :returns a tuple containing the two messages received"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(self._left_sensor.recv_sig)
            futureR = executor.submit(self._right_sensor.recv_sig)
            sigL = futureL.result()
            sigR = futureR.result()
        return sigL, sigR

    def _create_io_thread(self):
        """Methods which creates and starts the IO thread."""
        self._kill_thread_event.clear()
        self._io_thread = Thread(target=self._read_stereo_frames_in_background, args=())
        self._io_thread.start()

    def _kill_io_thread(self):
        """Methods which kills the IO thread."""
        self._kill_thread_event.set()
        self._io_thread.join()

    def _read_stereo_frames_in_background(self):
        """Methods which is meant to be executed by an IO thread: it repeatedly reads in background a pair of frames
        from the two sensors and stores it."""
        while not self._kill_thread_event.is_set():
            with ThreadPoolExecutor(max_workers=2) as executor:
                futureL = executor.submit(self._left_sensor.recv_frame)
                futureR = executor.submit(self._right_sensor.recv_frame)
                tstampL, frameL = futureL.result()
                tstampR, frameR = futureR.result()
            left_right_delay = tstampL - tstampR
            print(f'\rLatency: {time.time() - tstampL:.3f} s --- Left-Right Delay: {left_right_delay:.3f} s', end='')
            self._stereo_frames = frameL, frameR
            self._data_ready_event.set()

    def _recv_stereo_frames(self) -> (np.ndarray, np.ndarray):
        """Method which reads from the IO thread the most recent pair of stereo frames.
            :returns a tuple containing the two frames"""
        if not self._data_ready_event.wait(timeout=1.0):
            raise TimeoutError('Timeout while reading from the sensors.')
        self._data_ready_event.clear()
        return self._stereo_frames

    def load_calib_params(self, calib_file: str):
        # Load calibration parameters from file
        calib_params = np.load(calib_file + '.npz')
        # Set object's calibration parameters
        self._set_calib_params(calib_params)

    def _set_calib_params(self, calib_params):
        # Update object's calibration parameters
        self._calib_params.update(calib_params)
        self._is_calibrated = True

    def _save_calib_params(self, calib_file: str):
        # Copy the dictionary and add a key-value pair representing the file path
        # (required by NumPy 'savez_compressed' function)
        calib_file_to_save = self._calib_params.copy()
        calib_file_to_save['file'] = calib_file
        # Save calibration parameters to file
        np.savez_compressed(**calib_file_to_save)

    def load_disp_params(self, disp_file: str):
        # Load disparity parameters from file
        disp_params = {k: int(v) for k, v in np.load(disp_file + '.npz').items() if k != 'file'}
        # Set object's disparity parameters
        self._set_disp_params(disp_params)

    def _set_disp_params(self, disp_params):
        # Update object's disparity parameters
        self._disp_params.update(disp_params)
        self._has_disparity_params = True

    def _save_disp_params(self, disp_file: str):
        # Copy the dictionary and add a key-value pair representing the file path
        # (required by NumPy 'savez_compressed' function)
        disp_file_to_save = self._disp_params.copy()
        disp_file_to_save['file'] = disp_file
        # Save disparity parameters to file
        np.savez_compressed(**disp_file_to_save)

    def flush_pending_stereo_frames(self):
        """Method that concurrently flushes the pending frames of both ImageReceiver objects."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self._left_sensor.flush_pending_frames)
            executor.submit(self._right_sensor.flush_pending_frames)

    def close(self):
        """Method that closes the sockets and the contexts of both ImageSender objects to free resources."""
        self._left_sensor.close()
        self._right_sensor.close()

    def capture_sample_images(self, img_folder: str):
        """Method which captures sample stereo images
            :param img_folder: string representing the path to the folder in which images will be saved"""
        print('Collecting images of a chessboard for calibration...')

        # Initialize variables for countdown
        n_pics, tot_pics = 0, 30
        n_sec, tot_sec = 0, 4
        str_sec = '4321'

        # Define folders where calibration images will be stored
        pathL = os.path.join(img_folder, 'L')
        pathR = os.path.join(img_folder, 'R')

        # Wait for ready signal from sensors
        sigL, sigR = self.multicast_recv_sig()
        print(f'Left sensor: {sigL}')
        print(f'Right sensor: {sigR}')
        print('Both sensors are ready')

        # Synchronize sensors with a start signal
        self.multicast_send_sig(b'START')
        self._create_io_thread()

        # Save start time
        start_time = time.time()
        while True:
            # Get frames from both cameras
            frameL, frameR = self._recv_stereo_frames()
            # Flip frames horizontally to make it more comfortable for humans
            flipped_frameL = cv2.flip(frameL, 1)
            flipped_frameR = cv2.flip(frameR, 1)

            # Display counter on screen before saving frame
            if n_sec < tot_sec:
                # Draw on screen the current remaining pictures
                cv2.putText(img=flipped_frameL,
                            text=f'{n_pics}/{tot_pics}',
                            org=(int(10), int(40)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=3,
                            lineType=cv2.LINE_AA)
                # Draw on screen the current remaining seconds
                cv2.putText(img=flipped_frameR,
                            text=str_sec[n_sec],
                            org=(int(10), int(40)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=3,
                            lineType=cv2.LINE_AA)

                # If time elapsed is greater than one second, update 'n_sec'
                time_elapsed = time.time() - start_time
                if time_elapsed >= 1:
                    n_sec += 1
                    start_time = time.time()
            else:
                # When countdown ends, save original grayscale image to file
                gray_frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
                gray_frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(pathL, f'{n_pics:02d}' + '.jpg'), gray_frameL)
                cv2.imwrite(os.path.join(pathR, f'{n_pics:02d}' + '.jpg'), gray_frameR)
                # Update counters
                n_pics += 1
                n_sec = 0

                print(f'\n{n_pics}/{tot_pics} images collected.')

            # Display side by side the flipped frames
            frames = np.hstack((flipped_frameR, flipped_frameL))
            cv2.imshow('Left and right frames', frames)

            # If 'q' is pressed, or enough images are collected,
            # termination signal is sent to the sensors and streaming ends
            if (cv2.waitKey(1) & 0xFF == ord('q')) or n_pics == tot_pics:
                self.multicast_send_sig(b'STOP')
                self._kill_io_thread()
                print('\nStreaming ended.')
                break

        print('Images collected.')
        cv2.destroyAllWindows()
        self.flush_pending_stereo_frames()
        print('Pending frames flushed.')

    def calibrate(self,
                  img_folder: str,
                  pattern_size: Tuple[int, int],
                  square_length: float,
                  calib_file: str):
        """Computes the calibration parameters of a camera by using several pictures of a chessboard
            :param img_folder: string representing the path to the folder in which images are saved
            :param pattern_size: size of the chessboard used for calibration
            :param square_length: float representing the length, in mm, of the square edge
            :param calib_file: path to the file where calibration parameters will be saved"""
        # Define folders where calibration images will be stored
        pathL = os.path.join(img_folder, 'L')
        pathR = os.path.join(img_folder, 'R')
        # Get a list of images captured, one per folder
        img_namesL = glob.glob(os.path.join(pathL, '*.jpg'))
        img_namesR = glob.glob(os.path.join(pathR, '*.jpg'))
        # If one of the two lists is empty, raise exception
        if len(img_namesL) == 0 or len(img_namesR) == 0:
            raise CalibrationImagesNotFoundError(img_folder)
        # Produce a list of pairs '(right_image, left_image)'
        # If one list has more elements than the other, the extra elements will be automatically discarded by 'zip'
        img_name_pairs = list(zip(img_namesL, img_namesR))

        # Get number of available cores
        n_procs = psutil.cpu_count(logical=False)
        # Process in parallel stereo images
        stereo_img_names = []
        stereo_img_points_list = []
        stereo_img_drawn_corners_list = []
        with ProcessPoolExecutor(max_workers=n_procs) as executor:
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
                    print(f'No chessboard found in image {e.file}')

        # If no chessboard was detected, raise exception
        if len(stereo_img_points_list) == 0:
            raise ChessboardNotFoundError(img_folder)

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
        print('Calibrating left and right sensors...')
        with ProcessPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(cv2.calibrateCamera, obj_points, img_pointsL, (w, h), None, None)
            futureR = executor.submit(cv2.calibrateCamera, obj_points, img_pointsR, (w, h), None, None)
            rmsL, cam_mtxL, distL, _, _ = futureL.result()
            rmsR, cam_mtxR, distR, _, _ = futureR.result()

        print(f'Left sensor calibrated, RMS = {rmsL:.5f}')
        print(f'Right sensor calibrated, RMS = {rmsR:.5f}')

        # Use intrinsic parameters to calibrate more reliably the stereo camera
        print('Calibrating stereo camera...')
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
        print(f'Stereo camera calibrated, error: {error:.5f}')
        rot_mtxL, rot_mtxR, proj_mtxL, proj_mtxR, disp_to_depth_mtx, valid_ROIL, valid_ROIR = cv2.stereoRectify(
            cam_mtxL, distL, cam_mtxR, distR, (w, h), rot_mtx, trasl_mtx
        )
        # Compute the undistorted and rectify mapping
        mapxL, mapyL = cv2.initUndistortRectifyMap(cam_mtxL, distL, rot_mtxL, proj_mtxL, (w, h), cv2.CV_32FC1)
        mapxR, mapyR = cv2.initUndistortRectifyMap(cam_mtxR, distR, rot_mtxR, proj_mtxR, (w, h), cv2.CV_32FC1)

        # Save all camera parameters to .npz files
        calib_params = dict(cam_mtxL=cam_mtxL,
                            cam_mtxR=cam_mtxR,
                            disp_to_depth_mtx=disp_to_depth_mtx,
                            distL=distL,
                            distR=distR,
                            mapxL=mapxL,
                            mapxR=mapxR,
                            mapyL=mapyL,
                            mapyR=mapyR,
                            proj_mtxL=proj_mtxL,
                            proj_mtxR=proj_mtxR,
                            rot_mtx=rot_mtx,
                            rot_mtxL=rot_mtxL,
                            rot_mtxR=rot_mtxR,
                            trasl_mtx=trasl_mtx,
                            valid_ROIL=valid_ROIL,
                            valid_ROIR=valid_ROIR)
        self._set_calib_params(calib_params)
        self._save_calib_params(calib_file)
        print('Calibration parameters saved to file')

        # Plot the images with corners drawn on the chessboard and with calibration applied
        for i in range(0, len(stereo_img_names)):
            stereo_img_drawn_corners = stereo_img_drawn_corners_list[i]
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.suptitle(stereo_img_names[i])
            # Plot the original images
            ax[0][0].imshow(stereo_img_drawn_corners[0])
            ax[0][0].set_title('L frame')
            ax[0][1].imshow(stereo_img_drawn_corners[1])
            ax[0][1].set_title('R frame')

            # Remap images using the mapping found after calibration
            dstL = cv2.remap(stereo_img_drawn_corners[0], mapxL, mapyL, cv2.INTER_NEAREST)
            dstR = cv2.remap(stereo_img_drawn_corners[1], mapxR, mapyR, cv2.INTER_NEAREST)

            # Plot the undistorted images
            ax[1][0].imshow(dstL)
            ax[1][0].set_title('L frame undistorted')
            ax[1][1].imshow(dstR)
            ax[1][1].set_title('R frame undistorted')

            plt.show()

    def undistort_rectify(self, frameL: np.ndarray, frameR: np.ndarray) -> (np.ndarray, np.ndarray):
        # Check calibration data
        if not self._is_calibrated:
            raise MissingParametersError('Calibration')
        # Undistort and rectify using calibration data
        dstL = cv2.remap(frameL, self._calib_params['mapxL'], self._calib_params['mapyL'], cv2.INTER_LINEAR)
        dstR = cv2.remap(frameR, self._calib_params['mapxR'], self._calib_params['mapyR'], cv2.INTER_LINEAR)

        return dstL, dstR

    def disp_map_tuning(self, disp_file: str):
        """Allows to tune the disparity map
            :param disp_file: path to the file where disparity parameters will be saved"""
        print('Disparity map tuning...')

        # Check calibration data
        if not self._is_calibrated:
            raise MissingParametersError('Calibration')

        # Wait for ready signal from sensors
        sigL, sigR = self.multicast_recv_sig()
        print(f'Left sensor: {sigL}')
        print(f'Right sensor: {sigR}')
        print('Both sensors are ready')

        # Initialize variables for countdown
        n_sec, tot_sec = 0, 4
        str_sec = '4321'

        # Synchronize sensors with a start signal
        self.multicast_send_sig(b'START')

        # Save start time
        start_time = time.time()

        while True:
            # Get frames from both cameras and apply camera corrections
            frameL, frameR = self._recv_stereo_frames()
            dstL, dstR = self.undistort_rectify(frameL, frameR)
            # Flip frames horizontally to make it more comfortable for humans
            flipped_dstL = cv2.flip(dstL, 1)
            flipped_dstR = cv2.flip(dstR, 1)

            if n_sec < tot_sec:
                # Draw on screen the current remaining seconds
                cv2.putText(img=flipped_dstR,
                            text=str_sec[n_sec],
                            org=(int(10), int(40)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=3,
                            lineType=cv2.LINE_AA)

                # If time elapsed is greater than one second, update 'n_sec'
                time_elapsed = time.time() - start_time
                if time_elapsed >= 1:
                    n_sec += 1
                    start_time = time.time()
            else:
                print()
                break

            # Display side by side the frames
            frames = np.hstack((flipped_dstR, flipped_dstL))
            cv2.imshow('Left and right frames', frames)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        # When countdown ends, streaming is stopped, sockets are flushed and last frames are kept
        self.multicast_send_sig(b'STOP')
        self.flush_pending_stereo_frames()
        print('Pending frames flushed.')

        # Downsize
        dstL = cv2.resize(dstL, (480, 360))
        dstR = cv2.resize(dstR, (480, 360))

        # Create named window and sliders for tuning
        window_label = 'Disparity tuning'
        MDS_label = 'Minimum Disparity'
        MDS_label_neg = 'Minimum Disparity (negative)'
        NOD_label = 'Number of Disparities'
        SWS_label = 'SAD window size'
        PFC_label = 'PreFilter Cap'
        D12MD_label = 'Disp12MaxDiff'
        UR_label = 'Uniqueness Ratio'
        SPWS_label = 'Speckle Window Size'
        SR_label = 'Speckle Range'
        M_label = 'Mode'
        cv2.namedWindow(window_label)
        cv2.createTrackbar(MDS_label, window_label, 0, 40, lambda *args: None)
        cv2.createTrackbar(MDS_label_neg, window_label, 0, 40, lambda *args: None)
        cv2.createTrackbar(NOD_label, window_label, 0, 256, lambda *args: None)
        cv2.createTrackbar(SWS_label, window_label, 1, 15, lambda *args: None)
        cv2.createTrackbar(PFC_label, window_label, 0, 100, lambda *args: None)
        cv2.createTrackbar(D12MD_label, window_label, 0, 300, lambda *args: None)
        cv2.createTrackbar(UR_label, window_label, 0, 20, lambda *args: None)
        cv2.createTrackbar(SPWS_label, window_label, 0, 300, lambda *args: None)
        cv2.createTrackbar(SR_label, window_label, 0, 5, lambda *args: None)
        cv2.createTrackbar(M_label, window_label, 0, 1, lambda *args: None)

        while True:
            # Retrieve values set by trackers
            MDS = cv2.getTrackbarPos(MDS_label, window_label)
            MDS_neg = cv2.getTrackbarPos(MDS_label_neg, window_label)
            if MDS == 0:
                MDS = -MDS_neg
            # Convert NOD to next multiple of 16
            NOD = cv2.getTrackbarPos(NOD_label, window_label)
            NOD = NOD - (NOD % 16) + 16
            # Convert SWS to next odd number
            SWS = cv2.getTrackbarPos(SWS_label, window_label)
            SWS = SWS - (SWS % 2) + 2
            P1 = 8 * 3 * SWS ** 2
            P2 = 32 * 3 * SWS ** 2
            D12MD = cv2.getTrackbarPos(D12MD_label, window_label)
            UR = cv2.getTrackbarPos(UR_label, window_label)
            SPWS = cv2.getTrackbarPos(SPWS_label, window_label)
            SR = cv2.getTrackbarPos(SR_label, window_label)
            PFC = cv2.getTrackbarPos(PFC_label, window_label)
            M = cv2.getTrackbarPos(M_label, window_label)

            # Create and configure left and right stereo matchers
            stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=MDS,
                numDisparities=NOD,
                blockSize=SWS,
                P1=P1,
                P2=P2,
                disp12MaxDiff=D12MD,
                uniquenessRatio=UR,
                speckleWindowSize=SPWS,
                speckleRange=SR,
                preFilterCap=PFC,
                mode=M
            )

            # Compute disparity map
            disp = compute_disparity(dstL, dstR, stereo_matcher)
            # Apply colormap to disparity
            disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_PLASMA)

            # Stack resized frames and disparity map and display them
            disp_tune = np.hstack((dstL, disp_color))
            cv2.imshow(window_label, disp_tune)

            # If 'q' is pressed, exit and return parameters
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('../disp-samples/rect_dstL.jpg', dstL)
                cv2.imwrite('../disp-samples/rect_dstR.jpg', dstR)
                disp_params = {'MDS': MDS,
                               'NOD': NOD,
                               'SWS': SWS,
                               'D12MD': D12MD,
                               'UR': UR,
                               'SPWS': SPWS,
                               'SR': SR,
                               'PFC': PFC,
                               'M': M}
                break
        cv2.destroyAllWindows()
        self._set_disp_params(disp_params)
        self._save_disp_params(disp_file)
        print('Disparity parameters saved to file')

    def realtime_disp_map(self):
        """Displays a real-time disparity map"""
        print('Displaying real-time disparity map...')
        # Load calibration and disparity data
        if not self._is_calibrated:
            raise MissingParametersError('Calibration')
        if not self._has_disparity_params:
            raise MissingParametersError('Disparity')
        P1 = 8 * 3 * self._disp_params['SWS'] ** 2
        P2 = 32 * 3 * self._disp_params['SWS'] ** 2
        # Create and configure left and right stereo matchers
        stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self._disp_params['MDS'],
            numDisparities=self._disp_params['NOD'],
            blockSize=self._disp_params['SWS'],
            P1=P1,
            P2=P2,
            disp12MaxDiff=self._disp_params['D12MD'],
            uniquenessRatio=self._disp_params['UR'],
            speckleWindowSize=self._disp_params['SPWS'],
            speckleRange=self._disp_params['SR'],
            preFilterCap=self._disp_params['PFC'],
            mode=self._disp_params['M']
        )

        # Compute valid ROI
        valid_ROI = cv2.getValidDisparityROI(roi1=tuple(self._calib_params['valid_ROIL']),
                                             roi2=tuple(self._calib_params['valid_ROIR']),
                                             minDisparity=self._disp_params['MDS'],
                                             numberOfDisparities=self._disp_params['NOD'],
                                             blockSize=self._disp_params['SWS'])

        # Define skin color bounds in YCbCr color space
        skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        skin_upper = np.array([255, 173, 127], dtype=np.uint8)

        # Wait for ready signal from sensors
        sigL, sigR = self.multicast_recv_sig()
        print(f'Left sensor: {sigL}')
        print(f'Right sensor: {sigR}')
        print('Both sensors are ready')

        # Synchronize sensors with a start signal
        self.multicast_send_sig(b'START')

        disp_hist = []
        while True:
            # Get frames from both cameras and apply camera corrections
            frameL, frameR = self._recv_stereo_frames()
            dstL, dstR = self.undistort_rectify(frameL, frameR)

            # Crop frames to valid ROI
            x, y, w, h = valid_ROI
            dstL = dstL[y:y + h, x:x + w]
            dstR = dstR[y:y + h, x:x + w]

            # Downsize
            dstL = cv2.resize(dstL, (480, 360))
            dstR = cv2.resize(dstR, (480, 360))

            # Compute disparity map
            disp = compute_disparity(dstL, dstR, stereo_matcher, self._disp_bounds)
            # Denoise disparity map by averaging along temporal dimension
            # if len(disp_hist) == 2:
            #    disp = cv2.fastNlMeansDenoisingMulti([disp_hist[0], disp, disp_hist[1]], imgToDenoiseIndex=1,
            #                                          temporalWindowSize=3, templateWindowSize=1)
            #    disp_hist.pop(0)
            # disp_hist.append(disp)

            # Apply colormap to disparity
            disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_PLASMA)

            # Segment hand
            # Step 1: threshold disparity map
            _, disp_mask = cv2.threshold(disp, 185, 255, cv2.THRESH_BINARY)
            # Step 2: convert frame to YCbCr color space and segment pixels in the given range
            converted = cv2.cvtColor(dstL, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(converted, skin_lower, skin_upper)
            # Step 3: refine masks
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # disp_mask = cv2.morphologyEx(disp_mask, cv2.MORPH_OPEN, kernel)
            # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            # Step 4: apply both masks on the frame
            mask = np.bitwise_and(skin_mask, disp_mask)
            hand = cv2.bitwise_and(dstL.astype(np.uint8), dstL.astype(np.uint8), mask=mask)
            hand_disp = cv2.bitwise_and(disp, disp, mask=mask)
            # Step 5: apply close operator to refine the segmented image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel)
            hand_disp = cv2.morphologyEx(hand_disp, cv2.MORPH_CLOSE, kernel)

            # Display frames and disparity maps
            frames = np.hstack((dstL, dstR))
            cv2.imshow('Left and right frame', frames)
            cv2.imshow('Disparity', np.hstack((np.repeat(np.expand_dims(disp, axis=-1), 3, axis=-1), disp_color)))
            cv2.imshow("Hand", np.hstack((hand, np.repeat(np.expand_dims(hand_disp, axis=-1), 3, axis=-1))))

            # When 'q' is pressed, save current frames and disparity maps to file and break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('../disp-samples/Stereo_image.jpg', frames)
                cv2.imwrite('../disp-samples/Disparity.jpg', disp)
                # cv2.imwrite('../disp-samples/Hand.jpg', hand)
                self.multicast_send_sig(b'STOP')
                print()
                break
        cv2.destroyAllWindows()
        self.flush_pending_stereo_frames()
        print('Pending frames flushed.')
