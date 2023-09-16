import numpy as np
from src import V_x, V_y

def calculate_result_point(O1_new, K1_new, KSC_new, e3, coords_plane2, e_point_x, e_point_y, e_point_z):
    for it in range(min(len(coords_plane2), len(e_point_x))):
        #Координаты точки(проекции фокальной плоскости на землю) в ИСК
        e_point = np.array([e_point_x[it], e_point_y[it], e_point_z[it]])
        dir_pl = O1_new - K1_new
        dir_pl /= np.linalg.norm(dir_pl)
        #Коэффициент D из уравнения фокальной плоскости
        D_coefficient = -np.dot(dir_pl, K1_new)
        #Направляющий вектор
        dir = O1_new - e_point
        dir = dir / np.linalg.norm(dir)
        tmp = -(np.dot(e_point, dir_pl) + D_coefficient) / (np.dot(dir, dir_pl))

        #Координаты точки в ИСК
        e_point = e_point + tmp * dir
        #Координаты точки В фокальной плоскости
        V = np.linalg.inv(KSC_new.transpose()).dot(e_point) - np.linalg.inv(KSC_new.transpose()).dot(K1_new)

        #вектор сдвига пластины
        #delta_k1 = np.dot(np.linalg.inv(KSC_new.transpose()), K1_new - K1) - np.dot(np.linalg.inv(KSC_new).transpose(), K1_new)
        #для поля смещений
        V_x.append(V[0])
        V_y.append(V[1])

        #для поля скоростей
        #V_x.append((V[0] - coords_plane2[it][0]) / delta_t)# - 0.000001 * v_l[0])
        #V_y.append((V[1] - coords_plane2[it][1]) / delta_t)# - 0.000001 * v_l[1])