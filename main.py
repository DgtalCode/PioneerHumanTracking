import cv2
import time
import numpy as np
#from pioneer_sdk import Pioneer
from piosdk import Pioneer

useIntegratedCam = False

if not useIntegratedCam:
    pioneer = Pioneer()

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

heightMode = True

inWidth = 368
inHeight = 368
threshold = 0.3

last_seen = None

takePhoto = False
takePhotoTime = 0

nPoints = 18
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
    try:
        (a, b) = JOINTS_LIST[name]
        v = np.array([points[b][0] - points[a][0],
                      -(points[b][1] - points[a][1])])
        return v
    except:
        return None


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
    if real_value >= value - accuracy and real_value <= value + accuracy:
        return True


def not_none(*args):
    """ приверят, не является ли неизвестным хотя бы один из переданных аргументов """
    for arg in args:
        if arg is None:
            return False
    return True


protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"

if useIntegratedCam:
    cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

DEVICE = "gpu"

if DEVICE == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif DEVICE == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

np.set_printoptions(threshold=np.inf)

old_time = 0

while True:
    #print()

    t = time.time()
    if useIntegratedCam:
        # получение изображения с вебкамеры
        hasFrame, frame = cap.read()
    else:
        # получение изображения с коптера
        frame = cv2.imdecode(np.frombuffer(pioneer.get_raw_video_frame(), dtype=np.uint8), cv2.IMREAD_COLOR)

    frameCopy = np.copy(frame)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~ ФОТОГРАФИЯ БЕЗ НАДПИСЕЙ И ПРОЧЕГО ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    if takePhoto and time.time() - takePhotoTime > 3:
        cv2.imwrite('./imgae.jpg', frame)
        print("Photo taken at {}".format(time.ctime()))
        takePhoto = False
        if not useIntegratedCam:
            pioneer.led_control(255, 0, 0, 0)
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # конвертим входное изображение под необходимые для нейросети размеры
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # лист для хранения найденных кейпоинтов
    points = []

    for i in range(nPoints):
        # карта уверенности нейросети
        probMap = output[0, i, :, :]

        # поиск максимумов на карте
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # преобразование координат с выхода нейросети к изначальной системе
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # добавляем кейпоинт в массив, если уверенность в нем больше определенного порога
            points.append((int(x), int(y)))
        else:
            points.append(None)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~ ОТРИСОВКА СКЕЛЕТА ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    for pair in JOINTS_LIST.values():
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~ ПРОВЕРКА ЖЕСТОВ ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    # создание векторов, описывающих части тела
    right_forearm = vec("left_forearm")
    right_arm = vec("left_arm")
    left_forearm = vec("right_forearm")
    left_arm = vec("right_arm")

    # поиск углов в сочленениях
    right_elbow_alpha = angle(right_arm, right_forearm)
    right_arm_alpha = angle(right_arm, [100, 0])
    left_elbow_alpha = angle(left_arm, left_forearm)
    left_arm_alpha = angle(left_arm, [-100, 0])

    # массив для хранения распознанных действий
    # (необходим, чтобы не происходило детекта больше, чем одного жеста)
    action = []

    # выводим на экран все углы
    cv2.putText(frame, str(right_arm_alpha), (450, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 50, 255), 2,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, str(right_elbow_alpha), (450, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 50, 255), 2,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, str(left_arm_alpha), (450, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 50, 255), 2,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, str(left_elbow_alpha), (450, 350), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 50, 255), 2,
                lineType=cv2.LINE_AA)

    if points[2] is not None and points[5] is not None:
        cv2.putText(frame, str(abs(points[2][0] - points[5][0])), (150, 350), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 50, 255), 2, lineType=cv2.LINE_AA)

    if near(right_arm_alpha, 0) and near(right_elbow_alpha, -90) and right_forearm[1] > 0:
        action.append(1)

    if near(right_arm_alpha, 0) and near(right_elbow_alpha, -15):
        action.append(2)

    if near(left_arm_alpha, 0) and near(left_elbow_alpha, 90) and left_forearm[1] > 0:
        action.append(3)

    if near(left_arm_alpha, 0) and near(left_elbow_alpha, 15):
        action.append(4)

    if near(left_arm_alpha, 80, 20) and near(right_arm_alpha, -80, 20) and \
            near(left_elbow_alpha, 145, 20) and near(right_elbow_alpha, -145, 20) and \
            right_arm[0] > left_arm[0]:
        action.append(5)

    if near(left_arm_alpha, 0, 20) and near(right_arm_alpha, 0, 20) and \
            near(left_elbow_alpha, 90, 20) and near(right_elbow_alpha, -90, 20) and \
            right_forearm[1] < 0 and left_forearm[1] < 0:
        action.append(6)

    # если распознан только один жест, то выводим его на экран
    # иначе говорим. что ничего не найдено
    if len(action) == 1:
        cv2.putText(frame, "G %i" % action[0], (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 50, 255), 2,
                    lineType=cv2.LINE_AA)
    else:
        cv2.putText(frame, "G %i" % (-1), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 50, 255), 2, lineType=cv2.LINE_AA)

    cv2.imshow('kek', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~ УПРАВЛЕНИЕ КВАДРОКОПТЕРОМ ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    print(pioneer.get_task_id())
    if pioneer.get_task_id() == 0:
        key = cv2.waitKey(1)
        #newCommand = False

        if points[1] is not None:
            last_seen = points[1]

        # space
        if key == 32:
            pioneer.arm()
            pioneer.takeoff()
            flying = True
        # esc
        elif key == 27:
            pioneer.land()
            flying = False
        elif key == ord('k') and not flying:
            break
        elif key == ord('m'):
            manually = not manually

        if time.time() - old_time > 0.1:
            if not newCommand:
                if key == ord('w') or (len(action) == 1 and action[0] == 1):
                    cordY += speedXYZ
                    newCommand = True
                elif key == ord('s') or (len(action) == 1 and action[0] == 3):
                    cordY -= speedXYZ
                    newCommand = True
                if key == ord('d') or (len(action) == 1 and action[0] == 2):
                    cordX += speedXYZ
                    newCommand = True
                elif key == ord('a') or (len(action) == 1 and action[0] == 4):
                    cordX -= speedXYZ
                    newCommand = True

                if key == ord('r'):
                    cordZ += speedXYZ
                    newCommand = True
                elif key == ord('f'):
                    cordZ -= speedXYZ
                    newCommand = True

                if key == ord('q'):
                    yaw += speedYaw
                    newCommand = True
                elif key == ord('e'):
                    yaw -= speedYaw
                    newCommand = True

                if len(action) == 1 and action[0] == 5 and not takePhoto:
                    takePhoto = True
                    takePhotoTime = time.time()
                    print("Be ready!")
                    if not useIntegratedCam:
                        pioneer.led_control(255, 255, 0, 0)

                if len(action) == 1 and action[0] == 6:
                    pioneer.land();
                    flying = False

            if manually and flying and newCommand:
                old_time = time.time()
                if not useIntegratedCam:
                    pioneer.go_to_local_point(x=cordX, y=cordY, z=cordZ, yaw=yaw)
                newCommand = False

            elif not manually and last_seen is not None:
                newCommand = False
                old_time = time.time()
                v1 = [0, -100]
                v2 = [frameWidth // 2 - last_seen[0], -last_seen[1]]

                yaw_err = frameWidth//2 - last_seen[0]
                yaw += yaw_kp * yaw_err + yaw_kd * (yaw_err - yaw_errold)
                yaw_errold = yaw_err

                if heightMode:
                    z_err = frameHeight//2 - last_seen[1]
                    cordZ += z_kp * z_err + z_kd * (z_err - z_errold)
                    z_errold = z_err
                else:
                    y_err = frameHeight//2 - last_seen[1]
                    cordY = y_kp * y_err + y_kd * (y_err - y_errold)
                    y_errold = y_err

                pioneer.go_to_local_point(x=cordX, y=cordY, z=cordZ, yaw=np.radians(yaw))
    elif pioneer.get_task_id() == 1:
        pioneer.takeoff()
    elif pioneer.get_task_id() == 2:
        pioneer.land()
    elif pioneer.get_task_id() == 3:
        pioneer.go_to_local_point_as_loop()


cap.release()
cv2.destroyAllWindows()
