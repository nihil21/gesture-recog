import unittest
import numpy as np
from model.network_agent import ImageSender
from model.stereo_camera import StereoCamera
from typing import Tuple

IP_ADDR: str = 'localhost'
L_PORT: int = 8000
R_PORT: int = 8001
RES: Tuple[int, int] = (300, 300)


class TestStereoCamera(unittest.TestCase):

    def setUp(self):
        self.senderL = ImageSender(L_PORT, RES)
        self.senderR = ImageSender(R_PORT, RES)

        hostL = dict(ip_addr=IP_ADDR, port=L_PORT)
        hostR = dict(ip_addr=IP_ADDR, port=R_PORT)
        self.stereo = StereoCamera(hostL, hostR)

    def tearDown(self):
        self.senderL.close()
        self.senderR.close()
        self.stereo.close()

    def test_multicast_send_recv(self):
        msg = 'test'
        self.stereo.multicast_send(msg)
        resL = self.senderL.recv_msg()
        resR = self.senderR.recv_msg()
        self.assertEqual(resL, msg)
        self.assertEqual(resR, msg)

        msgL = 'testL'
        msgR = 'testR'
        self.senderL.send_msg(msgL)
        self.senderR.send_msg(msgR)
        res = self.stereo.multicast_recv()
        self.assertEqual(res['L'], msgL)
        self.assertEqual(res['R'], msgR)

    def test_recv_array(self):
        arrayL = np.random.rand(*RES)
        arrayR = np.random.rand(*RES)
        self.senderL.send_frame(arrayL)
        self.senderR.send_frame(arrayR)
        res = self.stereo._recv_stereo_frames()
        self.assertIsNone(np.testing.assert_almost_equal(res['L'], arrayL))
        self.assertIsNone(np.testing.assert_almost_equal(res['R'], arrayR))
