import cv2
import numpy as np


class Camera:
    """

    Control CSI Camera Jetson Nano

    """
    def __init__(self,
                 sensor_id=0,
                 capture_width=1280,
                 capture_height=720,
                 display_width=1280,
                 display_height=720,
                 framerate=30,
                 flip_method=0):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.framerate = framerate
        self.flip_method = flip_method

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.frame = np.empty((self.display_height, self.display_width, 3), dtype=np.uint8)

        try:
            self.cap = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        except:
            self.stop()
            raise RuntimeError("Error while initializing camera")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = frame
            return frame
        else:
            raise RuntimeError("Error while reading a frame")

    def stop(self):
        self.cap.release()

    def _gstreamer_pipeline(self):
        return ("nvarguscamerasrc sensor-id=%d !"
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (self.sensor_id,
                   self.capture_width,
                   self.capture_height,
                   self.framerate,
                   self.flip_method,
                   self.display_width,
                   self.display_height
                   )
                )
