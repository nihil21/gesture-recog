import unittest
import numpy as np
from model.network_agent import ImageSender, ImageReceiver
from typing import Tuple

IP_ADDR: str = 'localhost'
PORT: int = 8000
RES: Tuple[int, int] = (300, 300)


class TestNetworkAgent(unittest.TestCase):

    def setUp(self):
        self.sender = ImageSender(PORT, RES)
        self.receiver = ImageReceiver(IP_ADDR, PORT)

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def test_send_recv_msg(self):
        msg = 'test'
        self.sender.send_msg(msg)
        self.assertEqual(self.receiver.recv_msg(), msg)
        self.receiver.send_msg(msg)
        self.assertEqual(self.sender.recv_msg(), msg)

    def test_send_recv_array(self):
        array = np.random.rand(*RES)
        self.sender.send_frame(array)
        self.assertIsNone(np.testing.assert_almost_equal(self.receiver.recv_frame(), array))
