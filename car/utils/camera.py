import cv2


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
                 flip_method=0, ):
        self.running = True
        self.gstreamer = self.gstreamer_pipeline(sensor_id=sensor_id,
                                                 capture_width=capture_width,
                                                 capture_height=capture_height,
                                                 display_width=display_width,
                                                 display_height=display_height,
                                                 framerate=framerate,
                                                 flip_method=flip_method)
        self.cap = cv2.VideoCapture(self.gstreamer, cv2.CAP_GSTREAMER)
        assert self.cap.isOpened(), "Unable to open camera"

    def capture(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret is None:
                raise RuntimeError("Unable to get a frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

    def stop_capture(self):
        self.running = False
        self.cap.release()

    @staticmethod
    def gstreamer_pipeline(
            sensor_id,
            capture_width,
            capture_height,
            display_width,
            display_height,
            framerate,
            flip_method,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d !"
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    sensor_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

