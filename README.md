# gesture-recog
The aim of this project is to create a stereo camera able to recognize hand gestures.
### Hardware used
- Two Raspberry Pi boards.
- Two PiCamera modules (one per RP).

However, it's possible to run this program even on a computer with two webcams: in fact, if there is no PiCamera 
available, the system automatically falls back to webcam.
### Features:
1. Capture 30 pairs of images of a chessboard from the two PiCameras.
2. Calibrate the stereo camera using the images captured (server-side).
3. Compute the realtime disparity map.
4. [TODO] Detect hands and reproject them in a 3D space.
5. [TODO] Recognize hand gestures.

### Workflow:
1. Each Raspberry acts as a server.
2. The client connects to both servers.
3. The client asks to the user which action to perform, and tells each client to start streaming.
4. The images are sent from both Raspberries to the client via TCP socket.
5. The pairs of frames are received and processed concurrently.
