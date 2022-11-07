import Jetson.GPIO as GPIO


class DCMotor:
    """

    Control DC Motor Jetson Nano

    """
    def __init__(self):
        self.dc = 0

        GPIO.setmode(GPIO.TEGRA_SOC)
        self._ENA = "GPIO_PE6"
        self._IN1 = "DAP4_FS"
        self._IN2 = "SPI2_MOSI"

        GPIO.setup(self._ENA, GPIO.OUT, initial=GPIO.LOW)
        self.pwm = GPIO.PWM(self._ENA, 120)
        self.pwm.start(0)

        GPIO.setup(self._IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self._IN2, GPIO.OUT, initial=GPIO.LOW)

    def stop(self):
        GPIO.output(self._IN1, GPIO.LOW)
        GPIO.output(self._IN2, GPIO.LOW)

    def backward(self, speed):
        self._change_duty_cycle(speed)
        print(self.dc)
        GPIO.output(self._IN1, GPIO.HIGH)
        GPIO.output(self._IN2, GPIO.LOW)

    def forward(self, speed):
        self._change_duty_cycle(speed)
        GPIO.output(self._IN1, GPIO.LOW)
        GPIO.output(self._IN2, GPIO.HIGH)

    def stop_motor(self):
        self.pwm.stop()

    def _change_duty_cycle(self, dc):
        self.pwm.ChangeDutyCycle(dc)
        self.dc = dc
