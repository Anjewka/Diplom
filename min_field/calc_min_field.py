import math
from scipy import integrate
from src import coords_x, coords_y

#вычисления угла и вектора для минимизации поля скоростей
def calc_min_field(a0, b0, n0, m0, l0, coords_plane2, coords_2, V_x, V_y, V_x_2, V_y_2):
    for i in range(n0):
        for j in range(m0):
            coords_2[i][j] = coords_plane2[i * m0 + j]
            V_x_2[i][j] = V_x[i * m0 + j]
            V_y_2[i][j] = V_y[i * m0 + j]

    phi = 0
    u_x = 0
    u_y = 0
    #--------------------------------
    #метод симпсона используем scipy.integrate.simpson
    #для вычисления повторного интеграла
    for j in range(m0):
         coords_x.append(coords_2[0][j][0])
    for j in range(n0):
         coords_y.append(coords_2[j][0][1])

    Ip = [x for x in range(m0)]
    deltp = [x for x in range(n0)]
    phi = 0
    for j in range(m0):
        for k in range(n0):
            deltp[k] = (V_y_2[k][j] - coords_2[k][j][1]) * coords_2[k][j][0] - (V_x_2[k][j] - coords_2[k][j][0]) * coords_2[k][j][1]
        Ip[j] = integrate.simpson(deltp, coords_y)
    phi = 3 / (4*((a0) ** 3 * b0 + (b0)**3 * a0)) * integrate.simpson(Ip, coords_x)

    Ix = [x for x in range(n0)]
    deltx = [x for x in range(m0)]
    u_x = 0
    for j in range(n0):
        for k in range(m0):
            deltx[k] = V_x_2[j][k] - coords_x[k]
        Ix[j] = integrate.simpson(deltx, coords_x)
    u_x = 1 / (4 * a0 * b0) * integrate.simpson(Ix, coords_y)

    Iy = [x for x in range(m0)]
    delty = [x for x in range(n0)]
    u_y = 0
    for j in range(m0):
        for k in range(n0):
            delty[k] = V_y_2[k][j] - coords_y[k]
        Iy[j] = integrate.simpson(delty, coords_y)
    u_y = 1 / (4 * a0 * b0) * integrate.simpson(Iy, coords_x)


    # print("Оптимальный компенсирующий угол(градусы):", math.degrees(phi))
    # print("Компоненты вектора смещения центра ПЗС-линейки:", V_x[l0], V_y[l0])
    # print("Компоненты оптимального компенсирующего вектора:", u_x, u_y)
    return [phi, u_x, u_y]