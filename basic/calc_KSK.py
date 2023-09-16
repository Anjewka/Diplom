import numpy as np
import math
from basic.earth_inter import earth_inter

def calc_KSK(pitch_angle, roll_angle, yaw_angle, O1, OSC, r, d):    
    pitch_rot_matr = np.array([[math.cos(pitch_angle), 0, math.sin(pitch_angle)], [0, 1, 0], [-math.sin(pitch_angle), 0, math.cos(pitch_angle)]])   #тангаж
    roll_rot_matr = np.array([[1, 0, 0], [0, math.cos(roll_angle), -math.sin(roll_angle)], [0, math.sin(roll_angle), math.cos(roll_angle)]])    #крен
    yaw_rot_matr = np.array([[math.cos(yaw_angle), -math.sin(yaw_angle), 0], [math.sin(yaw_angle), math.cos(yaw_angle), 0], [0, 0, 1]])         #рыскание                           

    X = OSC.transpose().dot(pitch_rot_matr.dot(roll_rot_matr.dot(yaw_rot_matr.dot(np.array([1, 0, 0]))))) #направляющий вектор OX КСК в ОСК
    Y = OSC.transpose().dot(pitch_rot_matr.dot(roll_rot_matr.dot(yaw_rot_matr.dot(np.array([0, 1, 0]))))) #направляющий вектор OY КСК в ОСК
    Z = OSC.transpose().dot(pitch_rot_matr.dot(roll_rot_matr.dot(yaw_rot_matr.dot(np.array([0, 0, 1]))))) #направляющий вектор OZ КСК в ОСК

    dir = O1 / np.linalg.norm(O1)
    K1 = [O1[i] - d * dir[i] for i in range(3)]
    # print(K1)

    KSC = np.array([X, Y, Z]) #КСК
    # нормирование столбцов
    norms = np.linalg.norm(KSC, axis = 0)
    KSC = KSC / norms

    Z = Z / np.linalg.norm(Z)
    K1 = [O1[i] - d * Z[i] for i in range(3)]
    direc = (O1 - K1) / np.linalg.norm(O1 - K1)
    if earth_inter(O1, direc) == [0, 0]:
        print("BAD ANGEL")
    return K1, KSC