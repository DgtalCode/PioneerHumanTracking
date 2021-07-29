import cv2
import mediapipe as mp
import numpy as np
from piosdk import Pioneer
from collections import namedtuple
import time

# создание объектов для работы нейросети:
# для рисования
mpDrawings = mp.solutions.drawing_utils
# предварительный конфигуратор детектора
skeletonDetectorConfigurator = mp.solutions.pose
# создание детектора с некоторыми настройками
skDetector = skeletonDetectorConfigurator.Pose(static_image_mode=False,
                                               min_tracking_confidence=0.8,
                                               min_detection_confidence=0.8,
                                               model_complexity=2)

# использование встроенной камеры или камеры квадрокоптера
useIntegratedCam = False

# создание источников видео в зависимости от переменной
if not useIntegratedCam:
    pioneer = Pioneer()
else:
    cap = cv2.VideoCapture(0)

# объявление переменных, хранящих ширину и высоту изображения
IMGW, IMGH = None, None

# объявление переменной, хранящей значение нажатой кнопки на клавиатуре
key = -1

# объявление переменной, хранящей время начала отсчета таймера для фотографирования
# то есть текущее время на момент прихода команды на создание фотографии
take_photo_time = -1

# объявление переменной, хранящей время начала отсчета таймера для следующего детектирования жестов
# то есть текущее время на момент детектирования жеста, для создание небольшой задержки до след. срабатывания
pose_detected = -1

# переменные, хранящие положение квадрокоптера в пространстве
cordX = .0
cordY = .0
cordZ = 1.5
yaw = np.radians(0)

# шаг, на который будет происходить изменение положения квадрокоптера при выполнении команд
stepXY = 0.2

# переменные для работы ПД регулятора при повороте
yaw_err = 0
yaw_errold = 0
yaw_kp = .005
yaw_kd = .0025
yaw_k = 0.01

# переменные для работы ПД регулятора при движении вверх/вниз
z_err = 0
z_errold = 0
z_kp = .00004
z_kd = .00001

# переменные для работы ПД регулятора при движении вперед/назад
y_err = 0
y_errold = 0
y_kp = .12
y_kd = .01

# имена частей тела с индексами точек, образующих их
JOINTS_LIST = {"neck": [33, 0],
               "left_clavicle": [33, 12],
               "left_arm": [12, 14],
               "left_forearm": [14, 16],
               "right_clavicle": [33, 11],
               "right_arm": [11, 13],
               "right_forearm": [13, 15]}

# массив Точка имеет 4 именованных параметра, описывающих точку скелета
Point = namedtuple('Point', 'x y z visibility')

# массив, содержащий сгенерированные части тела в виде векторов,
# в котором к элементам можно обратиться через точку
Parts = namedtuple("Parts", JOINTS_LIST.keys())

# массив, описывающий конкретную часть тела в виде вектора
Part = namedtuple("Part", 'x y angle')


# объявление функции remap
def remap(oldValue, oldMin, oldMax, newMin, newMax):
    """
    Функция для преобразования числовых значений из одного диапазона в другой
    """
    oldRange = (oldMax - oldMin)
    if (oldRange == 0):
        newValue = newMin
    else:
        newRange = (newMax - newMin)
        newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
    return newValue
# конец объявления

# объявление функции convert_points
def convert_points(points):
    """
    Функция для конвертации определенных нейросетью точек скелета
    из относительных координат (0-1.0) в абсолютные координаты
    """
    # массив, в котором будут храниться конвертированные точки
    converted_points = []

    # генерация базовой точки (между лопатками)
    # х - (х левой лопатки + х правой лопатки) / 2
    # у - (у левой лопатки + у правой лопатки) / 2
    # остальное по аналогии
    base_point = Point(x=round(IMGW * (points[12].x + points[11].x) // 2),
                       y=round(IMGH * (points[12].y + points[11].y) // 2),
                       z=(points[12].z + points[11].z) / 2,
                       visibility=(points[12].visibility + points[11].visibility) / 2)
    # непосредственная конвертация координат точек путем умножения
    # относительной координаты на ширину(высоту) изображения
    # (z и видимость остаются без изменений)
    for p in points:
        converted_points.append(Point(x=round(IMGW * p.x),
                                      y=round(IMGH * p.y),
                                      z=p.z,
                                      visibility=p.visibility))
    converted_points.append(base_point)
    return converted_points
# конец объявления


# объявление функции ang
def ang(v1):
    """
    Функция рассчитывает направление вектора на плоскости и возвращает угол от 0 до 359
    """
    angle = round(np.degrees(np.arctan2(v1[1], -v1[0])))
    angle = remap(angle, -180, 179, 0, 359)
    return round(angle)
# конец объявления


# объявление функции generate_parts_vectors
def generate_parts_vectors(pts):
    """
    Функция для представления частей тела в виде векторов.
    Принимает набор точек, а возвращает вектора.
    """
    j = {}
    # проход по элементам словаря с именами и точками частей тела
    for joint in JOINTS_LIST.items():
        # 2 точки, образующие часть тела
        pos = joint[1]
        # из переданного набора найденных точек скелета выбираются эти 2 и считается вектор
        vec_x = pts[pos[1]].x - pts[pos[0]].x
        vec_y = pts[pos[1]].y - pts[pos[0]].y
        # сохранение вектора с именем части тела
        j.update({
            joint[0]: Part(vec_x, vec_y, ang([vec_x, vec_y]))
        })
    # конвертация в массив, к элементам которого можно обращаться через точку
    j = Parts(**j)
    return j
# конец объявления


# объявление функции eq
def eq(num1, num2, err=10):
    """
    функция для сравнивания двух чисел с погрешностью
    """
    if num1 < 0 or num2 < 0:
      return True
    return True if abs(num1-num2) <= err else False
# конец объявления


# объявление функции eq_all
def eq_all(lside=[], rside=[], neck=[]):
    """
    функция для быстрого сравнения всех прописанных векторов
    """
    ans = True
    if lside:
        ans = eq(parts.left_clavicle.angle, lside[0]) and \
              eq(parts.left_arm.angle, lside[1]) and \
              eq(parts.left_forearm.angle, lside[2]) and \
              ans
    if rside:
        ans = eq(parts.right_clavicle.angle, rside[0]) and \
              eq(parts.right_arm.angle, rside[1]) and \
              eq(parts.right_forearm.angle, rside[2]) and \
              ans
    if neck:
        ans = eq(parts.neck.angle) and \
              ans
    return ans
# конец объявления


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ГЛАВНЫЙ ЦИКЛ~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# объявление массива для хранения конвертированных точек
converted_points = []

# если используется не интегрированная камера
if not useIntegratedCam:
    # включение двигателей коптера
    pioneer.arm()
    # взлет
    pioneer.takeoff()

while True:
    # считывание кадра либо с веб-камеры, либо с коптера
    if useIntegratedCam:
        ret, frame = cap.read()
        if not ret:
            continue
    else:
        img = pioneer.get_raw_video_frame()
        frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    # отзеркаливание изображения
    frame = cv2.flip(frame, 1)

    # получение размеров изображения
    IMGW = np.shape(frame)[1]
    IMGH = np.shape(frame)[0]

    # определение точек скелета
    detected_skeletons = skDetector.process(frame)

    # проверка, найдены ли точки и находится ли коптер в воздухе
    if detected_skeletons.pose_landmarks is not None:
        # запись всех точек в переменную с более коротким именем
        points = detected_skeletons.pose_landmarks.landmark

        # конвертация координат точек из относительных в абсолютные
        # (из диапазона от 0 до 1 в диапазон от 0 до ширины/высоты)
        converted_points = convert_points(points)
        # представление частей тела в виде векторов
        parts = generate_parts_vectors(converted_points)

        # отрисовка базовой точки (между лопатками)
        cv2.circle(frame, (converted_points[33].x, converted_points[33].y), 4, (255,0,0), 3)

        # проверка, был ли найден жест
        # необходима для создания задержки (промежутка) между детектированием
        if pose_detected == -1:
            # проверка направлений векторов и выявление поз
            if eq_all(lside=[180, 270, 45], rside=[0, 270, 135]):
                print("POSE 1")
                if take_photo_time == -1:
                    print("Picture will be saved in 5 seconds")
                    take_photo_time = time.time()
                pose_detected = time.time()

            elif eq_all(lside=[180, 180, 180]):
                print("POSE 2")
                cordX += stepXY
                pose_detected = time.time()

            elif eq_all(rside=[0, 0, 0]):
                print("POSE 3")
                cordX -= stepXY
                pose_detected = time.time()

            elif eq_all(lside=[180, 180, 90]):
                print("POSE 4")
                cordY += stepXY
                pose_detected = time.time()

            elif eq_all(rside=[0, 0, 90]):
                print("POSE 5")
                cordY -= stepXY
                pose_detected = time.time()

            elif eq_all(lside=[180, 225, 270], rside=[0, 315, 270]):
                print("POSE 6")
                pose_detected = time.time()
                # если используется НЕ встроенная в компьютер камера
                if not useIntegratedCam:
                    pioneer.land()

        # если с момента обнаружения жеста прошла секунда - сбросить переменную, блокирующую детектирование
        if pose_detected != -1 and time.time() - pose_detected > 1:
            pose_detected = -1

        # если время вызова таймера существует и уже прошло больше 5 секунд, то сохранить фото
        if take_photo_time != -1 and time.time() - take_photo_time > 5:
            cv2.imwrite("image", frame)
            take_photo_time = -1
            pose_detected = -1

    # если используется НЕ встроенная в компьютер камера
    if not useIntegratedCam:
        # если сконвертированные точки существуют, то работают регуляторы:
        if converted_points:
            # регулятор для удержания человека в центре изображения по рысканью (вращение вокруг своей оси)
            yaw_err = -(IMGW // 2 - converted_points[33].x) * yaw_k
            yaw_u = yaw_kp * yaw_err - yaw_kd * (yaw_err - yaw_errold)
            yaw_errold = yaw_err

            # регулятор определенного расстояния до человека по оси Y (вперед/назад)
            y_err = -(-0.15 - converted_points[33].z)
            y_u = y_kp * y_err - y_kd * (y_err - y_errold)
            y_errold = y_err

            # регулятор для удержания человека в центре изображения по оси Z (вверх/вниз)
            z_err = (IMGH // 2 - converted_points[33].y)
            z_u = z_kp * z_err - z_kd * (z_err - z_errold)
            z_errold = z_err

            # обновление переменных, содержащих значения (координаты) для удержания коптером
            yaw += yaw_u
            cordY += y_u
            cordZ += z_u
            pioneer.go_to_local_point(cordX, cordY, cordZ, yaw=yaw)

    # отрисовка всех точек и линий средствами используемой библиотеки
    mpDrawings.draw_landmarks(frame, detected_skeletons.pose_landmarks,
                              skeletonDetectorConfigurator.POSE_CONNECTIONS)

    # создание окна с изображением
    cv2.imshow("frame", frame)

    # считывание идентификатора нажатой кнопки с клавиатуры
    key = cv2.waitKey(1)

    # выход из программы при нажатии кнопки q
    if key == ord('q'):
        # если используется НЕ встроенная в компьютер камера
        if not useIntegratedCam:
            pioneer.disarm()
        break

    # посадка при нажатии кнопки l
    if key == ord('l'):
        # если используется НЕ встроенная в компьютер камера
        if not useIntegratedCam:
            pioneer.command_id = 0
            pioneer.land()
            cordX, cordY = 0, 0
            cordZ = 1.5

    # взлет при нажатии кнопки j
    if key == ord('j'):
        # если используется НЕ встроенная в компьютер камера
        if not useIntegratedCam:
            pioneer.command_id = 0
            pioneer.arm()
            pioneer.takeoff()

# завершение работы захвата изображений с камеры
if useIntegratedCam:
    cap.release()
# закрытие всех окон
cv2.destroyAllWindows()

