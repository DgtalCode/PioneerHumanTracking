import cv2
import mediapipe as mp
import numpy as np
from piosdk import Pioneer
from SkeletonPosePredictor import VecPosePredictor
import SkeletonPosePredictor as sk
import time

useIntegratedCam = True

if not useIntegratedCam:
    pioneer = Pioneer()
    img_shape = (640, 480)
else:
    cap = cv2.VideoCapture(0)
    img_shape = (1920, 1080)

# predictor = VecPosePredictor('./data.json')

mpDrawings = mp.solutions.drawing_utils
skeletonDetectorConfigurator = mp.solutions.pose
skDetector = skeletonDetectorConfigurator.Pose(static_image_mode=False,
                                               min_tracking_confidence=0.5,
                                               min_detection_confidence=0.5, model_complexity=2)

key = -1

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

JOINTS_LIST = {"neck": [33, 0],
               "left_clavicle": [33, 12],
               "left_arm": [12, 14],
               "left_forearm": [14, 16],
               "right_clavicle": [33, 11],
               "right_arm": [11, 13],
               "right_forearm": [13, 15]}

predictor = VecPosePredictor(JOINTS_LIST)

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ГЛАВНЫЙ ЦИКЛ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

frame = None

predict = False
predict_time = time.time()

while True:
    # считывание кадра либо с веб-камеры, либо с пионера
    if useIntegratedCam:
        ret, frame = cap.read()
        if not ret:
            continue
    else:
        img = pioneer.get_raw_video_frame()
        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)

    IMG_WIDTH = np.shape(frame)[1]
    IMG_HEIGHT = np.shape(frame)[0]

    frame.flags.writeable = False
    detected_skeletons = skDetector.process(frame)
    frame.flags.writeable = True

    if detected_skeletons.pose_landmarks is not None:
        points = detected_skeletons.pose_landmarks.landmark

        base_point = sk.Point(x=(points[12].x + points[11].x) / 2,
                              y=(points[12].y + points[11].y) / 2,
                              z=(points[12].z + points[11].z) / 2,
                              visibility=(points[12].visibility + points[11].visibility) / 2)

        converted_points = predictor.convert_points(points)

        joints = predictor.generate_joints_vectors(converted_points)

        ra1 = predictor.angle_between(joints.right_clavicle, joints.right_arm)
        ra2 = predictor.angle_between(joints.right_arm, joints.right_forearm)
        la1 = predictor.angle_between(joints.left_clavicle, joints.left_arm)
        la2 = predictor.angle_between(joints.left_arm, joints.left_forearm)
        print(base_point.z)

        if key == ord('0'):
            predictor.start_writing(0)
        if key == ord('1'):
            predictor.start_writing(1)
        if key == ord('2'):
            predictor.start_writing(2)
        if key == ord('x'):
            predictor.save_data()
        # if key == ord('9'):
        #     predict = not predict
        # if predict and time.time() - predict_time > 1:
        #     a = predictor.predict([relative_points])
        #     print(a)
        #     predict_time = time.time()

    mpDrawings.draw_landmarks(frame, detected_skeletons.pose_landmarks,
                              skeletonDetectorConfigurator.POSE_CONNECTIONS)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

