import time
import math
import smbus


class Servo:
    """

    Control Servo Jetson Nano

    """
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __MODE2 = 0x01
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        result = self.bus.read_byte_data(self.address, reg)
        return result

    def set_pwm_freq(self, freq):
        prescaleval = 25000000.0  # 25MHz
        prescaleval /= 4096.0  # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0

        prescale = math.floor(prescaleval + 0.5)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self.write(self.__MODE1, newmode)  # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)
        self.write(self.__MODE2, 0x04)

    def set_pwm(self, channel, on, off):
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)

    def set_servo_pulse(self, channel, pulse):
        pulse = pulse * 4096 / 20000  # PWM frequency is 50HZ,the period is 20000us
        self.set_pwm(channel, 0, int(pulse))

    def set_rotation_angle(self, channel, angle):
        if 0 <= angle <= 180:
            temp = angle * (2000 / 180) + 501
            self.set_servo_pulse(channel, temp)
        else:
            print("Angle out of range")

    def exit_servo(self):
        self.write(self.__MODE2, 0x00)

