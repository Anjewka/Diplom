import numpy as np
import math

def fsk_dots(fsk_center, FSK, e_point_x, e_point_y, e_point_z, O1):
    earth_dots = []
    for it in range(len(e_point_x)):
        #Координаты точки в ФСК
        fsk_dot = np.linalg.inv(FSK.transpose()).dot(np.array([e_point_x[it], e_point_y[it], e_point_z[it]])) - np.linalg.inv(FSK.transpose()).dot(fsk_center)
        # fsk_dot = FSK.dot(np.array([e_point_x[it], e_point_y[it], e_point_z[it]])) + fsk_center
        fsk_dot[2] = math.sqrt((O1[0] - e_point_x[it]) ** 2 + (O1[1] - e_point_y[it]) ** 2 + (O1[2] - e_point_z[it]) ** 2)
        earth_dots.append(fsk_dot.tolist())
    return earth_dots