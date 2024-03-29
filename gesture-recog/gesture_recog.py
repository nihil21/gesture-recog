import argparse
import zmq
from stereo_camera.stereo_camera import StereoCamera
from stereo_camera.errors import (
    CalibrationImagesNotFoundError,
    ChessboardNotFoundError,
    MissingParametersError,
)
import typing

# Ports of the sensors
IMG_PORT: int = 8000
CTRL_PORT: int = 8001

# Folders to store images for calibration
IMG_FOLDER: str = "../calibration-images/"

# Chessboard size
PATTERN_SIZE: typing.Tuple[int, int] = (8, 5)

# Square length
SQUARE_LEN: float = 26.5  # mm

# File where calibration data is stored
CALIB_FILE: str = "../calibration-data/calib"  # .npz

# File where disparity data is stored
DISP_FILE: str = "../calibration-data/disp"  # .npz

# Folder to store dataset images
DATA_FOLDER: str = "../dataset/"


def print_title():
    print(
        "  ________                 __                                __________                            "
    )
    print(
        " /  _____/  ____   _______/  |_ __ _________   ____          \\______   \\ ____   ____  ____   ____  "
    )
    print(
        "/   \\  ____/ __ \\ /  ___/\\   __\\  |  \\_  __ \\_/ __ \\   ______ |       _// __ \\_/ ___\\/  _ "
        "\\ / ___\\ "
    )
    print(
        "\\    \\_\\  \\  ___/ \\___ \\  |  | |  |  /|  | \\/\\  ___/  /_____/ |    |   \\  ___/\\  \\__"
        "(  <_> ) /_/  >"
    )
    print(
        " \\______  /\\___  >____  > |__| |____/ |__|    \\___  >         |____|_  /\\___  >\\___  >____/\\___  /"
    )
    print(
        "        \\/     \\/     \\/                          \\/                 \\/     \\/     \\/     /_____/  "
    )
    print("-" * 99)


def user_input() -> int:
    """Displays a menu of the available actions and asks user's input
    :return sel: integer representing user's choice"""

    print("")
    print("=" * 50)
    print("1 - Collect images for calibration")
    print("2 - Perform calibration")
    print("3 - Disparity map tuning on sample images")
    print("4 - Real time disparity map")
    print("5 - Exit")

    while True:
        try:
            sel = input("Select one of the options [1, 2, 3, 4, 5]: ")
            sel = int(sel)
            if sel not in [1, 2, 3, 4, 5]:
                print("The option inserted is not valid, retry.")
            else:
                break
        except ValueError:
            print("The option inserted is not numeric, retry.")
    print("-" * 50)
    return sel


def main():
    # Display the title of the tool in ASCII art
    print_title()

    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-iL", "--ip_addrL", required=True, help="hostname of the left sensor"
    )
    ap.add_argument(
        "-iR", "--ip_addrR", required=True, help="hostname of the right sensor"
    )
    args = vars(ap.parse_args())

    ip_addrL = args["ip_addrL"]
    ip_addrR = args["ip_addrR"]

    # Create StereoCamera object
    print("Local endpoints created:")
    print(f"\t{ip_addrL}:{IMG_PORT} for image streaming from left sensor;")
    print(f"\t{ip_addrR}:{IMG_PORT} for image streaming from right sensor;")
    print(f"\t{ip_addrL}:{CTRL_PORT} for control signals to left sensor;")
    print(f"\t{ip_addrR}:{CTRL_PORT} for control signals to right sensor.")

    stereo_camera = StereoCamera(ip_addrL, ip_addrR, IMG_PORT, CTRL_PORT)
    try:
        stereo_camera.load_calib_params(CALIB_FILE)
        print(
            f"Calibration parameters loaded from file {CALIB_FILE}, stereo camera is already calibrated."
        )
    except IOError:
        print(
            f"Could not load calibration parameters from file {CALIB_FILE}, "
            f"stereo camera must be calibrated before usage."
        )
    try:
        stereo_camera.load_disp_params(DISP_FILE)
        print(
            f"Disparity parameters loaded from file {DISP_FILE}, stereo camera has already the optimal parameters."
        )
    except IOError:
        print(
            f"Could not load disparity parameters from file {DISP_FILE}, "
            "optimal parameters must be tuned before usage."
        )

    try:
        while True:
            # Display action menu and ask for user input
            sel = user_input()

            # Invoke corresponding function
            if sel == 1:
                stereo_camera.multicast_send_sig(b"STREAM")
                stereo_camera.capture_sample_images(IMG_FOLDER)
            elif sel == 2:
                try:
                    stereo_camera.calibrate(
                        IMG_FOLDER, PATTERN_SIZE, SQUARE_LEN, CALIB_FILE
                    )
                except CalibrationImagesNotFoundError as e:
                    print(
                        f"There are no images to perform calibration in folder {e.folder}, collect them first."
                    )
                except ChessboardNotFoundError as e:
                    print(
                        f"No chessboards were detected in the images provided in folder {e.file}, "
                        f"capture better images."
                    )
            elif sel == 3:
                stereo_camera.multicast_send_sig(b"STREAM")
                try:
                    stereo_camera.disp_map_tuning(DISP_FILE)
                except MissingParametersError as e:
                    print(f"{e.parameter_cat} parameters missing.")
            elif sel == 4:
                stereo_camera.multicast_send_sig(b"STREAM")
                stereo_camera.realtime_disp_map()
            elif sel == 5:
                break
    except zmq.ZMQError:
        print("\nError communicating over the network.")
    except TimeoutError as e:
        print(f"{e}: the sensor may be disconnected or too slow.")
    except KeyboardInterrupt:
        print("\nTermination enforced manually.")
    finally:
        # Free resources
        stereo_camera.close()
        print("Terminating...")


if __name__ == "__main__":
    main()
