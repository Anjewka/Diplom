import math
import numpy as np
from basic.earth_inter import earth_inter
from src import w_z

def calculate_earth_point(O1, K1, KSC, coords_plane2, e_point_x, e_point_y, e_point_z, delta_t):
    for it in range(len(coords_plane2)):
        x_coordinate = coords_plane2[it][0]
        y_coordinate = coords_plane2[it][1]

        #Координаты точки в КСК
        coords_plane3 = np.array([x_coordinate, y_coordinate, 0])
        point_in_KSC = KSC.transpose().dot(coords_plane3) + K1
        
        dir = O1 - point_in_KSC
        dir = dir / np.linalg.norm(dir)
        tmp = min(earth_inter(point_in_KSC, dir)[0], earth_inter(point_in_KSC, dir)[1])
        e_p = point_in_KSC + tmp * dir
        if delta_t == 0:
            e_point_x.append(e_p[0])
            e_point_y.append(e_p[1])
            e_point_z.append(e_p[2])
        if tmp != 0 and delta_t !=0:
            #Угол поворота земли за время экспозиции(или другое заданное время)
            α_1 = w_z * delta_t

            e_point_x.append(math.cos(α_1) * e_p[0] - math.sin(α_1) * e_p[1])
            e_point_y.append(math.sin(α_1) * e_p[0] + math.cos(α_1) * e_p[1])
            e_point_z.append(e_p[2])