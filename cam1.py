import threading
import time

import cv2
import numpy as np
import socket


class Camera:

    def __init__(self, timeout=0.5, ip='192.168.4.1', port=8888, video_buffer_size=65000, log_connection=True):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.VIDEO_BUFFER_SIZE = video_buffer_size
        self.tcp = None
        self.udp = None
        self._video_frame_buffer = None
        self.raw_video_frame = None
        self.connected = None
        self.log_connection = log_connection
        self._thread_stop = threading.Event()
        self._thread_stop.set()

        for i in range(15):
            if not self.connected:
                self.connected = self.reconnect()
                if self.log_connection:
                    print('Camera CONNECTED')
                break
            else:
                if self.log_connection:
                    print('Camera DISCONNECTED')
        self.connect()
        time.sleep(2)

    def new_tcp(self):
        """Returns new TCP socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        return sock

    def new_udp(self):
        """Returns new UDP socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        return sock

    def connect(self):
        self._getting_frames_tread = threading.Thread(target=self._getting_frames, daemon=True)
        self._thread_stop.clear()
        self._getting_frames_tread.start()

    def disconnect(self):
        """Disconnect."""
        self._thread_stop.set()
        self.connected = False
        if self.tcp is not None:
            self.tcp.close()
            self.tcp = None
        if self.udp is not None:
            self.udp.close()
            self.udp = None
        if self.log_connection:
            print('Camera DISCONNECTED')

    def reconnect(self):
        """Connect to TCP and UDP sockets. Creates new ones if necessary."""
        if self.tcp is not None:
            self.tcp.close()
            self.tcp = None
        if self.udp is not None:
            self.udp.close()
            self.udp = None
        self.tcp = self.new_tcp()
        self.udp = self.new_udp()
        try:
            self.tcp.connect((self.ip, self.port))
            self.udp.bind(self.tcp.getsockname())
        except:
            return False
        return True

    def _getting_frames(self):
        while True:
            if self._thread_stop.is_set():
                break
            if not self.connected:
                while not self.reconnect():
                    pass
                self.connected = True
                if self.log_connection:
                    print('Camera CONNECTED')
            try:
                self._video_frame_buffer = self.udp.recv(self.VIDEO_BUFFER_SIZE)
                end = self._video_frame_buffer.rfind(b'\xff\xd9')
                if end == -1:
                    continue
                self._video_frame_buffer = self._video_frame_buffer[:end + 2]
                beginning = self._video_frame_buffer.rfind(b'\xff\xd8')
                if beginning == -1:
                    continue
                self.raw_video_frame = self._video_frame_buffer[beginning:]
            except:
                if self.connected:
                    self.connected = False
                    if self.log_connection:
                        print('Camera DISCONNECTED')

    def get_frame(self):
        """
        Returns raw frame (bytes).
        :return: raw_frame or None
        """
        return self.raw_video_frame

    def get_cv_frame(self):
        """
        Returns decoded frame.
        :return: cv_frame or None
        """

        frame = self.get_frame()
        if frame is not None:
            frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame


class VideoStream:
    def __init__(self, logger=True):
        self.camera = Camera(log_connection=logger)
        self.logger = logger
        self._vidio_stream = None
        self._stop = threading.Event()
        self._stop.set()

    def start(self):
        if not self._stop.is_set():
            return
        self._stop.clear()
        self._vidio_stream = threading.Thread(target=self._stream, daemon=True)
        self._vidio_stream.start()

    def stop(self):
        self._stop.set()

    def _stream(self):
        """
        Continuously receives JPEG frames from ESP32, parses them, and renders it using standard OpenCV's image rendering
        facilities.
        """
        keymap = {"esc": 27}
        while True:
            key = cv2.waitKey(1)
            if key == keymap["esc"] or self._stop.is_set():
                cv2.destroyAllWindows()
                break

            # Try to receive a frame
            camera_frame = self.camera.get_frame()

            if camera_frame is None:
                if self.logger:
                    print("No frame")
                continue

            # Decode and render JPEG frame
            camera_frame = cv2.imdecode(np.frombuffer(camera_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('pioneer_camera_stream', camera_frame)
