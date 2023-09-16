import math
import numpy as np
from basic.earth_inter import earth_inter
from src import w_z

#находим центр и направляющие векторы ФСК в ИСК,
def fsk(O1, K1, KSC, a0, b0):
    fsk_center_plate = KSC.transpose().dot([0, 0, 0]) + K1
    fsk_x_plate = KSC.transpose().dot([-b0, 0, 0]) + K1
    fsk_y_plate = KSC.transpose().dot([0, -a0, 0]) + K1
        
    dir = O1 - fsk_center_plate
    dir = dir / np.linalg.norm(dir)
    tmp = min(earth_inter(fsk_center_plate, dir)[0], earth_inter(fsk_center_plate, dir)[1])
    fsk_center = fsk_center_plate + tmp * dir

    dir = O1 - fsk_x_plate
    dir = dir / np.linalg.norm(dir)
    tmp = min(earth_inter(fsk_x_plate, dir)[0], earth_inter(fsk_x_plate, dir)[1])
    fsk_x = fsk_x_plate + tmp * dir

    dir = O1 - fsk_y_plate
    dir = dir / np.linalg.norm(dir)
    tmp = min(earth_inter(fsk_y_plate, dir)[0], earth_inter(fsk_y_plate, dir)[1])
    fsk_y = fsk_y_plate + tmp * dir

    fsk_ox = []
    for i in range(3):
        fsk_ox.append(fsk_x[i] - fsk_center[i])
    fsk_ox = fsk_ox / np.linalg.norm(fsk_ox)

    fsk_oy = []
    for i in range(3):
        fsk_oy.append(fsk_y[i] - fsk_center[i])
    fsk_oy = fsk_oy / np.linalg.norm(fsk_oy)

    fsk_oz = []
    for i in range(3):
        fsk_oz.append(fsk_center[i])
    fsk_oz = fsk_oz / np.linalg.norm(fsk_oz)
    FSK = np.array([fsk_ox, fsk_oy, fsk_oz])
    norms = np.linalg.norm(FSK, axis=0)
    FSK = FSK / norms
    return fsk_center, FSK, fsk_x