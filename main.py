import cv2
import time
import mediapipe as mp
import numpy as np
from piosdk import Pioneer

useIntegratedCam = True

if not useIntegratedCam:
    pioneer = Pioneer()
else:
    cap = cv2.VideoCapture(0)

mpDrawings = mp.solutions.drawing_utils
skeletonDetectorConfigurator = mp.solutions.pose
skDetector = skeletonDetectorConfigurator.Pose(static_image_mode=False,
                                               min_tracking_confidence=0.7,
                                               min_detection_confidence=0.6, model_complexity=2)

cordX = .0
cordY = .0
cordZ = 1
speedXYZ = .2
yaw = np.radians(0)
speedYaw = np.radians(10)

yaw_err = 0
yaw_errold = 0
yaw_kp = .05
yaw_kd = .025

z_err = 0
z_errold = 0
z_kp = .0004
z_kd = .0001
cordZ1 = 1.0

y_err = 0
y_errold = 0
y_kp = .0004
y_kd = .0001

manually = True
flying = False

last_seen = None

takePhoto = False
takePhotoTime = 0

nPoints = 32
JOINTS_LIST = {"neck": [1, 0],
               "right_clavicle": [1, 2],
               "right_arm": [2, 3],
               "right_forearm": [3, 4],
               "left_clavicle": [1, 5],
               "left_arm": [5, 6],
               "left_forearm": [6, 7],
               "right_body": [1, 8],
               "right_hip": [8, 9],
               "right_calf": [9, 10],
               "left_body": [1, 11],
               "left_hip": [11, 12],
               "left_calf": [12, 13],
               "right_eye": [0, 14],
               "right_ear": [14, 16],
               "left_eye": [0, 15],
               "left_ear": [15, 17]}


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def sign(num):
    """ Возврат знака числа """
    return 1 if num >= 0 else -1


def vec(name):
    """ создает вектор на основе задетекченных точек по переданному имени """
    pass


def angle(vec_1, vec_2):
    """ считает угол между двумя векторами """
    if vec_1 is None or vec_2 is None:
        return None
    # скалярное произведение векторов / (длина вектора 1 * длина вектора 2)
    a = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    a = -(np.arccos(a) * sign(vec_1[0]))
    return np.degrees(a)


def near(real_value, value, accuracy=15):
    """ Считает, равны ли примерно переданные значения """
    if real_value is None:
        return False
    if value - accuracy <= real_value <= value + accuracy:
        return True


def not_none(*args):
    """ проверят, не является ли неизвестным хотя бы один из переданных аргументов """
    for arg in args:
        if arg is None:
            return False
    return True


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ГЛАВНЫЙ ЦИКЛ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

frame = None

while True:
    # считывание кадра либо с веб-камеры, либо с пионера
    if useIntegratedCam:
        ret, frame = cap.read()
        if not ret:
            continue
    else:
        img = pioneer.get_raw_video_frame()
        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)

    IMG_WIDTH = np.shape(frame)[1]
    IMG_HEIGHT = np.shape(frame)[0]

    frame.flags.writeable = False
    detected_skeletons = skDetector.process(frame)
    frame.flags.writeable = True

    if detected_skeletons.pose_landmarks is not None:
        points = detected_skeletons.pose_landmarks.landmark

    mpDrawings.draw_landmarks(frame, detected_skeletons.pose_landmarks,
                              skeletonDetectorConfigurator.POSE_CONNECTIONS)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
