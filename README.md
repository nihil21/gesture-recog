# gesture-recog
The aim of this project is to create a stereo camera able to recognize hand gestures.
### Hardware used
- Two Raspberry Pi boards.
- Two PiCamera modules (one per RP).
### Workflow:
1. The two RPs capture the Right and Left images.
2. The images are sent via TCP socket to the main computer.
3. The main computer uses them to calibrate the two cameras and compute a disparity map.
4. At this point, computer vision and machine learning techniques will be used to recognize hand gestures.
