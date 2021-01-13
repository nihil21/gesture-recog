import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from model.errors import *
from model.network_agent import ImageReceiver
from utils.image_proc_tools import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Union, Tuple


class StereoCamera:
    """Class representing a StereoCamera, composed by two remote ImageSender objects; it enables and simplifies the
    concurrent communication with both sensors.

    Attributes:
        left_sensor -- the ImageSender representing the left sensor
        right_sensor -- the ImageSender representing the right sensor
        calib_params --- dictionary containing the calibration parameters (initially empty)
        is_calibrated --- boolean variable indicating whether the stereo camera is calibrated or not
        (initially set to 'False')
        disp_params --- dictionary containing the disparity parameters (initially empty)
        has_disparity_params --- boolean variable indicating whether the stereo camera has disparity
        parameters set or not (initially set to 'False')

    Methods:
        multicast_send --- method which enables the concurrent sending of a string message via the
        two ImageReceiver objects
        multicast_recv --- method which enables the concurrent reception of a string message via the
        two ImageReceiver objects
        recv_stereo_frames --- method which enables the concurrent reception of a stereo OpenCV image via the
        two ImageReceiver objects
        load_calib_params --- method which reads the calibration parameters from a given file
        set_calib_params --- method which sets/updates the calibration parameters
        save_calib_params --- method which persists the calibration parameters to a given file
        load_disp_params --- method which reads the disparity parameters from a given file
        set_disp_params --- method which sets/updates the disparity parameters
        save_disp_params --- method which persists the disparity parameters to a given file
        flush_pending_stereo_frames --- method which flushes the pending stereo OpenCV images over the TCP sockets
        of the two ImageReceiver objects
        close --- method which releases the network resources used by the two ImageReceiver objects"""

    def __init__(self,
                 hostL: Dict[str, Union[str, int]],
                 hostR: Dict[str, Union[str, int]]):
        self.left_sensor = ImageReceiver(hostL['ip_addr'], hostL['port'])
        self.right_sensor = ImageReceiver(hostR['ip_addr'], hostR['port'])
        self.calib_params = dict()
        self.is_calibrated = False
        self.disp_params = dict()
        self.has_disparity_params = False

    def multicast_send(self, msg: str):
        """Method which enables the concurrent communication with both ImageSender objects
            :param msg: string representing the message to be sent concurrently to both ImageSender objects"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.left_sensor.send_msg, msg)
            executor.submit(self.right_sensor.send_msg, msg)

    def multicast_recv(self) -> Dict[str, str]:
        """Method which enables the concurrent reception of messages from both ImageSender objects
            :returns a tuple containing the two messages received"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(self.left_sensor.recv_msg)
            futureR = executor.submit(self.right_sensor.recv_msg)

            return dict(L=futureL.result(), R=futureR.result())

    def recv_stereo_frames(self) -> (np.ndarray, np.ndarray):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(self.left_sensor.recv_frame)
            futureR = executor.submit(self.right_sensor.recv_frame)

            return futureL.result(), futureR.result()

    def load_calib_params(self, calib_file: str):
        # Load calibration parameters from file
        calib_params = np.load(calib_file + '.npz')
        # Set object's calibration parameters
        self.set_calib_params(calib_params)

    def set_calib_params(self, calib_params):
        # Update object's calibration parameters
        self.calib_params.update(calib_params)
        self.is_calibrated = True

    def save_calib_params(self, calib_file: str):
        # Copy the dictionary and add a key-value pair representing the file path
        # (required by NumPy 'savez_compressed' function)
        calib_file_to_save = self.calib_params.copy()
        calib_file_to_save['file'] = calib_file
        # Save calibration parameters to file
        np.savez_compressed(**calib_file_to_save)

    def load_disp_params(self, disp_file: str):
        # Load disparity parameters from file
        disp_params = np.load(disp_file + '.npz')
        # Set object's disparity parameters
        self.set_disp_params(disp_params)

    def set_disp_params(self, disp_params):
        # Update object's disparity parameters
        self.disp_params.update(disp_params)
        self.has_disparity_params = True

    def save_disp_params(self, disp_file: str):
        # Copy the dictionary and add a key-value pair representing the file path
        # (required by NumPy 'savez_compressed' function)
        disp_file_to_save = self.disp_params.copy()
        disp_file_to_save['file'] = disp_file
        # Save disparity parameters to file
        np.savez_compressed(**disp_file_to_save)

    def flush_pending_stereo_frames(self):
        """Method that concurrently flushes the pending frames of both ImageReceiver objects"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.left_sensor.flush_pending_frames)
            executor.submit(self.right_sensor.flush_pending_frames)

    def close(self):
        """Method that closes the sockets and the contexts of both ImageSender objects to free resources"""
        self.left_sensor.close()
        self.right_sensor.close()

    def capture_sample_images(self, img_folder: str):
        """Function which captures sample stereo images
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
        res = self.multicast_recv()
        print('Left sensor: {0:s}'.format(res['L']))
        print('Right sensor: {0:s}'.format(res['R']))
        print('Both sensors are ready')

        # Synchronize sensors with a start signal
        self.multicast_send('start')

        # Save start time
        start_time = datetime.now()
        while n_pics < tot_pics:
            # Get frames from both cameras
            frameL, frameR = self.recv_stereo_frames()

            # Display counter on screen before saving frame
            if n_sec < tot_sec:
                # Draw on screen the current remaining seconds
                cv2.putText(img=frameL,
                            text=str_sec[n_sec],
                            org=(int(10), int(40)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=3,
                            lineType=cv2.LINE_AA)
                # Draw on screen the current remaining pictures
                cv2.putText(img=frameR,
                            text='{:d}/{:d}'.format(n_pics, tot_pics),
                            org=(int(10), int(40)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=3,
                            lineType=cv2.LINE_AA)

                # If time elapsed is greater than one second, update 'n_sec'
                time_elapsed = (datetime.now() - start_time).total_seconds()
                if time_elapsed >= 1:
                    n_sec += 1
                    start_time = datetime.now()
            else:
                # When countdown ends, save grayscale image to file
                gray_frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
                gray_frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(pathL, '{0:02d}'.format(n_pics) + '.jpg'), gray_frameL)
                cv2.imwrite(os.path.join(pathR, '{0:02d}'.format(n_pics) + '.jpg'), gray_frameR)
                # Update counters
                n_pics += 1
                n_sec = 0

                print('{0:d}/{1:d} images collected'.format(n_pics, tot_pics))

            # Display side by side the frames
            frames = np.hstack((frameL, frameR))
            cv2.imshow('Left and right frames', frames)

            # If 'q' is pressed, or enough images are collected,
            # termination signal is sent to the slaves and streaming ends
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.multicast_send('term')
                break
            if n_pics == tot_pics:
                self.multicast_send('term')

        cv2.destroyAllWindows()
        self.flush_pending_stereo_frames()
        print('Pending frames flushed')
        print('Images collected')

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
                    print('No chessboard found in image {0:s}'.format(e.file))

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
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureL = executor.submit(cv2.calibrateCamera, obj_points, img_pointsL, (w, h), None, None)
            futureR = executor.submit(cv2.calibrateCamera, obj_points, img_pointsR, (w, h), None, None)
            rmsL, cam_mtxL, distL, _, _ = futureL.result()
            rmsR, cam_mtxR, distR, _, _ = futureR.result()

        print('Left sensor calibrated, RMS = {0:.5f}'.format(rmsL))
        print('Right sensor calibrated, RMS = {0:.5f}'.format(rmsR))

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
        print('Stereo camera calibrated, error: {0:.5f}'.format(error))
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
        self.set_calib_params(calib_params)
        self.save_calib_params(calib_file)
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

    def disp_map_tuning(self, disp_file: str):
        """Allows to tune the disparity map
            :param disp_file: path to the file where disparity parameters will be saved"""
        print('Disparity map tuning...')

        # Load calibration data
        if not self.is_calibrated:
            raise MissingParametersError('Calibration')

        # Wait for ready signal from sensors
        res = self.multicast_recv()
        print('Left sensor: {0:s}'.format(res['L']))
        print('Right sensor: {0:s}'.format(res['R']))
        print('Both sensors are ready')

        # Initialize variables for countdown
        n_sec, tot_sec = 0, 4
        str_sec = '4321'

        # Synchronize sensors with a start signal
        self.multicast_send('start')

        # Save start time
        start_time = datetime.now()

        frameL, frameR = None, None
        while n_sec < tot_sec:
            # Get frames from both cameras
            frameL, frameR = self.recv_stereo_frames()

            # Draw on screen the current remaining seconds
            cv2.putText(img=frameL,
                        text=str_sec[n_sec],
                        org=(int(10), int(40)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA)

            # If time elapsed is greater than one second, update 'n_sec'
            time_elapsed = (datetime.now() - start_time).total_seconds()
            if time_elapsed >= 1:
                n_sec += 1
                start_time = datetime.now()

            # Display side by side the frames
            frames = np.hstack((frameL, frameR))
            cv2.imshow('Left and right frames', frames)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        # When countdown ends, streaming is stopped, sockets are flushed and last frames are kept
        self.multicast_send('term')
        self.flush_pending_stereo_frames()
        print('Pending frames flushed')

        # Create named window and sliders for tuning
        window_label = 'Disparity tuning'
        MDS_label = 'Minimum Disparity'
        MDS_label_neg = 'Minimum Disparity (negative)'
        NOD_label = 'Number of Disparities'
        SWS_label = 'SAD window size'
        PFC_label = 'PreFilter Cap'
        cv2.namedWindow(window_label)
        cv2.createTrackbar(MDS_label, window_label, 0, 40, lambda *args: None)
        cv2.createTrackbar(MDS_label_neg, window_label, 0, 40, lambda *args: None)
        cv2.createTrackbar(NOD_label, window_label, 0, 144, lambda *args: None)
        cv2.createTrackbar(SWS_label, window_label, 1, 15, lambda *args: None)
        cv2.createTrackbar(PFC_label, window_label, 0, 63, lambda *args: None)

        # Parameters not tuned
        D12MD = 1
        UR = 10
        SPWS = 7
        SR = 2

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
            P1 = 8 * SWS ** 2
            P2 = 32 * SWS ** 2
            PFC = cv2.getTrackbarPos(PFC_label, window_label)

            # Create and configure left and right stereo matchers
            stereo_matcherL = cv2.StereoSGBM_create(
                minDisparity=MDS,
                numDisparities=NOD,
                blockSize=SWS,
                P1=P1,
                P2=P2,
                disp12MaxDiff=D12MD,
                uniquenessRatio=UR,
                speckleWindowSize=SPWS,
                speckleRange=SR,
                preFilterCap=PFC
            )
            stereo_matcherR = cv2.ximgproc.createRightMatcher(stereo_matcherL)

            # Filter parameters
            LAMBDA = 80000
            SIGMA = 1.2
            # Create filter
            # noinspection PyUnresolvedReferences
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_matcherL)
            wls_filter.setLambda(LAMBDA)
            wls_filter.setSigmaColor(SIGMA)

            # Compute disparity map
            disp, disp_gray, dstL, dstR = compute_disparity(frameL, frameR,
                                                            self.calib_params['mapxL'],
                                                            self.calib_params['mapxR'],
                                                            self.calib_params['mapyL'],
                                                            self.calib_params['mapyR'],
                                                            stereo_matcherL, stereo_matcherR, wls_filter)

            # Stack resized frames and disparity map and display them
            disp_tune = np.hstack((dstL, disp))
            cv2.imshow(window_label, disp_tune)

            # If 'q' is pressed, exit and return parameters
            if cv2.waitKey(1) & 0xFF == ord('q'):
                disp_params = dict(MDS=MDS,
                                   NOD=NOD,
                                   SWS=SWS,
                                   D12MD=D12MD,
                                   UR=UR,
                                   SPWS=SPWS,
                                   SR=SR,
                                   PFC=PFC)
                break
        cv2.destroyAllWindows()
        self.set_disp_params(disp_params)
        self.save_disp_params(disp_file)
        print('Disparity parameters saved to file')

    def realtime_disp_map(self):
        """Displays a real-time disparity map"""
        print('Displaying real-time disparity map...')
        # Load calibration and disparity data
        if not self.is_calibrated:
            raise MissingParametersError('Calibration')
        if not self.has_disparity_params:
            raise MissingParametersError('Disparity')
        P1 = 8 * self.disp_params['SWS'] ** 2
        P2 = 32 * self.disp_params['SWS'] ** 2

        # Create and configure left and right stereo matchers
        stereo_matcherL = cv2.StereoSGBM_create(
            minDisparity=self.disp_params['MDS'],
            numDisparities=self.disp_params['NOD'],
            blockSize=self.disp_params['SWS'],
            P1=P1,
            P2=P2,
            disp12MaxDiff=self.disp_params['D12MD'],
            uniquenessRatio=self.disp_params['UR'],
            speckleWindowSize=self.disp_params['SPWS'],
            speckleRange=self.disp_params['SR'],
            preFilterCap=self.disp_params['PFC']
        )
        stereo_matcherR = cv2.ximgproc.createRightMatcher(stereo_matcherL)

        # Filter parameters
        LAMBDA = 80000
        SIGMA = 1.2
        # Create filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_matcherL)
        wls_filter.setLambda(LAMBDA)
        wls_filter.setSigmaColor(SIGMA)

        # Compute valid ROI
        valid_ROI = cv2.getValidDisparityROI(roi1=tuple(self.calib_params['valid_ROIL']),
                                             roi2=tuple(self.calib_params['valid_ROIR']),
                                             minDisparity=self.disp_params['MDS'],
                                             numberOfDisparities=self.disp_params['NOD'],
                                             blockSize=self.disp_params['SWS'])

        # Create MOG2 background subtractors for both frames
        backSubL = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
        backSubR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)

        # Wait for ready signal from sensors
        res = self.multicast_recv()
        print('Left sensor: {0:s}'.format(res['L']))
        print('Right sensor: {0:s}'.format(res['R']))
        print('Both sensors are ready')

        # Synchronize sensors with a start signal
        self.multicast_send('start')

        while True:
            # Get frames from both cameras
            frameL, frameR = self.recv_stereo_frames()

            # Compute disparity map
            disp, disp_gray, dstL, dstR = compute_disparity(frameL, frameR,
                                                            self.calib_params['mapxL'],
                                                            self.calib_params['mapxR'],
                                                            self.calib_params['mapyL'],
                                                            self.calib_params['mapyR'],
                                                            stereo_matcherL, stereo_matcherR, wls_filter, valid_ROI)

            # Compute foreground mask based on both frames and update background
            hand_maskL = backSubL.apply(dstL, learningRate=0.5)
            hand_maskR = backSubR.apply(dstR, learningRate=0.5)

            '''
            # Threshold function
            def distance_threshold_fn(p):
                depth = compute_depth(disp_point=p,
                                      baseline=abs(self.calib_params['trasl_mtx'][0][0]),
                                      alpha_uL=self.calib_params['cam_mtxL'][0][0],
                                      alpha_uR=self.calib_params['cam_mtxR'][0][0],
                                      u_0L=self.calib_params['cam_mtxL'][0][2],
                                      u_0R=self.calib_params['cam_mtxR'][0][2])
                if thresh[0] <= depth <= thresh[1]:
                    return 255
                else:
                    return 0
    
            # Converting distance_threshold_fn to Python's ufunc in order to improve performance
            distance_threshold_ufn = np.frompyfunc(distance_threshold_fn, 1, 1)
    
            # Segment disparity
            hand_mask = distance_threshold_ufn(disp_gray).astype(np.uint8)
            '''

            # Apply masks to frames
            handL = cv2.bitwise_and(dstL.astype(np.uint8), dstL.astype(np.uint8), mask=hand_maskL)
            handR = cv2.bitwise_and(dstR.astype(np.uint8), dstR.astype(np.uint8), mask=hand_maskR)

            # Determine contours of hands
            contoursL, _ = cv2.findContours(cv2.cvtColor(handL, cv2.COLOR_BGR2GRAY),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(handL, contoursL, -1, color=(0, 255, 0), thickness=cv2.FILLED)
            contoursR, _ = cv2.findContours(cv2.cvtColor(handR, cv2.COLOR_BGR2GRAY),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(handR, contoursR, -1, color=(0, 255, 0), thickness=cv2.FILLED)

            # Display frames and disparity maps
            frames = np.hstack((dstL, dstR))
            hand = np.hstack((handL, handR))
            cv2.imshow('Left and right frame', frames)
            cv2.imshow('Disparity', disp)
            cv2.imshow("Hand", hand)

            # When 'q' is pressed, save current frames and disparity maps to file and break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('../disp-samples/Stereo_image.jpg', frames)
                cv2.imwrite('../disp-samples/Disparity.jpg', disp)
                cv2.imwrite('../disp-samples/Hand.jpg', hand)
                self.multicast_send('term')
                break

        self.flush_pending_stereo_frames()
        print('Pending frames flushed')
        cv2.destroyAllWindows()
