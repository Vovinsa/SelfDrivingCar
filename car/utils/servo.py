import time
from adafruit_servokit import ServoKit
from adafruit_pca9685 import PCA9685
import board
import busio


class ServoMotor:
    """

    Control ServoMotor Jetson Nano

    """
    def __init__(self, channel):
        # self.RIGHT = 0
        # self.LEFT = 50
        # self.FORWARD = 25

        i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(i2c)
        self.motor = ServoKit(channels=16)
        self._set_actuation_range(50)
        self._set_pca_frequency(100)
        self._set_pca_duty_cycle(0xffff, channel=channel)

    def set_rotation_angle(self, channel, angle):
        # if angle > 50:
        #     angle = 50
        # elif angle < 0:
        #     angle = 0
        self.motor.servo[channel].angle = angle
        time.sleep(0.005)

    def _set_pca_frequency(self, freq):
        self._pca.frequency = freq

    def _set_pca_duty_cycle(self, dc, channel):
        ch = self._pca.channels[channel]
        ch.duty_cycle = dc

    def _set_actuation_range(self, act_range):
        self.motor.actuation_range = act_range
