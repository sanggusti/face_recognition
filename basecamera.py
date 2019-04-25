import threading
import time
from _thread import get_ident

TIMEOUT = 5

class CameraEvent(object):
    def __init__(self):
        self.event = {}

class BaseCamera():
    thread = None
    frame = None
    last_access = 0
    # event = CameraEvent()

    def __init__(self):
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        BaseCamera.last_access = time.time()
        return BaseCamera.frame

    @staticmethod
    def frames():
        raise RuntimeError("Needs to be implemented in a subclass")
    
    @classmethod
    def _thread(cls):
        frames_iter = cls.frames()
        for frames in frames_iter:
            BaseCamera.frame = frames
            time.sleep(0)
            if time.time() - BaseCamera.last_access > TIMEOUT:
                frames_iter.close()
                break
        BaseCamera.thread = None

class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # imgs = [cv2.imread('./images/'+i+'.jpg') for i in ['1','2','3']]
        # self.frames = imgs
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def getFrame(self):
        receive, img = self.cap.read()
        if receive:
            return cv2.imencode('.jpg',img)[1].tobytes()
        # retval, to_return = cv2.imencode('.jpg',self.frames[int(time.time()) % 3])
        # return to_return.tobytes()