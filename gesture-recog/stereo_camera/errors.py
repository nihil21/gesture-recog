class OutOfSyncError(Exception):
    """Exception raised when the two sensors go out of sync.

    Attributes:
        delay -- the detected delay between the two sensors"""

    def __init__(self, delay: float):
        self.delay = delay


class CalibrationImagesNotFoundError(Exception):
    """Exception raised when no calibration image is found in the given folder.

    Attributes:
        folder -- the path to the folder where calibration images should be found"""

    def __init__(self, folder: str):
        self.folder = folder


class ChessboardNotFoundError(Exception):
    """Exception raised when a chessboard is not found in a calibration image, or when it is not found in every
    calibration images available.

    Attributes:
        file -- the file generating the error"""

    def __init__(self, file: str):
        self.file = file


class MissingParametersError(Exception):
    """Exception raised when the essential parameters of a StereoCamera object are not computed but are required
    to perform a certain operation.

    Attributes:
        parameter_cat --- the type of parameters missing"""

    def __init__(self, parameter_cat: str):
        self.parameter_cat = parameter_cat
