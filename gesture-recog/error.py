class ChessboardNotFoundError(Exception):
    """Exception raised when a chessboard is not found in a calibration image.

    Attributes:
        image -- the image generating the error
    """

    def __init__(self, image):
        self.image = image
