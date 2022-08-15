import Jetson.GPIO as GPIO


class DCMotor:
    """

    Control DC Motor Jetson Nano

    """
    def __init__(self):
        self._ENA = "GPIO_PE6"
        self._IN1 = "DAP4_FS"
        self._IN2 = "SPI2_MOSI"
        GPIO.setmode(GPIO.TEGRA_SOC)
        GPIO.setup(self._ENA, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self._IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self._IN2, GPIO.OUT, initial=GPIO.LOW)

    def stop(self):
        GPIO.output(self._ENA, GPIO.HIGH)
        GPIO.output(self._IN1, GPIO.LOW)
        GPIO.output(self._IN2, GPIO.LOW)

    def backward(self):
        GPIO.output(self._IN1, GPIO.HIGH)
        GPIO.output(self._IN2, GPIO.LOW)

    def forward(self):
        GPIO.output(self._IN1, GPIO.LOW)
        GPIO.output(self._IN2, GPIO.HIGH)
