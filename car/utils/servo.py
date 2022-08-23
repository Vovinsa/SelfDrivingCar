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
        i2c = busio.I2C(board.SCL, board.SDA)
        self.channel = channel
        self._pca = PCA9685(i2c)
        self.motor = ServoKit(channels=16)
        self._set_actuation_range(50)
        self._set_pca_frequency(100)
        self._set_pca_duty_cycle(0xffff)

    def set_rotation_angle(self, angle):
        self.motor.servo[self.channel].angle = angle
        time.sleep(0.005)

    def _set_pca_frequency(self, freq):
        self._pca.frequency = freq

    def _set_pca_duty_cycle(self, dc):
        ch = self._pca.channels[self.channel]
        ch.duty_cycle = dc

    def _set_actuation_range(self, act_range):
        self.motor.actuation_range = act_range
