import numpy as np
import math
import matplotlib.pyplot as plt
from src import V_x_old, V_y_old, V_x_alpha, V_y_alpha, V_x_new_no_alpha, V_y_new_no_alpha, V_x_new, V_y_new

def speed_vector(a0, b0, n0, m0, n1, m1, l0, V_x, V_y, u_x, u_y, phi, coords_plane2, scale_prm):
    #print(coords_plane2)
    fig = plt.figure('Min field')
    ax = fig.add_subplot()

    ticks_x = np.linspace(-b0, b0, m0)
    ticks_y = np.linspace(-a0, a0, n0)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)

    for it in range(len(V_x)):
        V_x_old.append(V_x[it] - V_x[l0])
        V_y_old.append(V_y[it] - V_y[l0])
    #построение поля смещения до минимизации с учетом вычета вектора центра прямоугольника
    #масштабирование картинки
    len_coords_0 = math.sqrt((V_x_old[0] - coords_plane2[0][0]) ** 2 + (V_y_old[0] - coords_plane2[0][1]) ** 2)
    len_coords_1 = math.sqrt((V_x_old[m0-1] - coords_plane2[m0-1][0]) ** 2 + (V_y_old[m0-1] - coords_plane2[m0-1][1]) ** 2)
    len_coords_2 = math.sqrt((V_x_old[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][0]) ** 2 + (V_y_old[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][1]) ** 2)
    len_coords_3 = math.sqrt((V_x_old[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][0]) ** 2 + (V_y_old[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][1]) ** 2)
    len_coords = (len_coords_0 + len_coords_1 + len_coords_2 + len_coords_3) / 4
    #print(len_coords)

    for it in range(len(V_x)):
        ax.quiver(coords_plane2[it][0], coords_plane2[it][1], V_x_old[it] - coords_plane2[it][0], V_y_old[it] - coords_plane2[it][1],  angles = 'xy', scale_units = 'xy', width = 0.0045, scale = len_coords / scale_prm)#scale = len_coords / 0.008)#scale=0.00032 и 0.000042)
    ax.grid(color = 'green')
    ax.set_aspect('equal')
    ax.set_aspect(abs((b0/a0)))
    plt.xlim([-(b0 + 3 * m1), b0 + 3 * m1])
    plt.ylim([-(a0 + 3 * n1), a0 + 3 * n1])
    plt.show()

    #построение поля смещения после вычисления результата минимизации
    fig1 = plt.figure('Min field')
    ax1 = fig1.add_subplot()
    ticks_x = np.linspace(-b0, b0, m0)
    ticks_y = np.linspace(-a0, a0, n0)
    ax1.set_xticks(ticks_x)
    ax1.set_yticks(ticks_y)

    #также вычислим поле с учетом вычета вектора центра прямоугольника
    for it in range(len(V_x)):
        V_x_alpha.append(V_x[it] - V_x[l0])
        V_y_alpha.append(V_y[it] - V_y[l0])

    #также вычислим поле с учетом вычета оптимального компенсирующего вектора, но не учитывая опитмальный угол вращения ПЗС-линейки
    for it in range(len(V_x)):
        V_x_new_no_alpha.append(V_x[it] - u_x)
        V_y_new_no_alpha.append(V_y[it] - u_y)

    for it in range(len(V_x)):
        x = V_x[it]
        y = V_y[it]
        V_x[it] = x* math.cos(phi) + y * math.sin(phi)
        V_y[it] = -x * math.sin(phi) + y * math.cos(phi)

    #также вычислим поле с учетом вычета вектора центра прямоугольника  и поворота на оптимальный угол вращения ПЗС-линейки
    for it in range(len(V_x)):
        V_x_alpha.append(V_x[it] - V_x[l0])
        V_y_alpha.append(V_y[it] - V_y[l0])

    for it in range(len(V_x)):
        V_x_new.append(V_x[it] - u_x)
        V_y_new.append(V_y[it] - u_y)

    print('V(0, 0) = ', math.sqrt(V_x[len(V_x) - 1] ** 2 + V_y[0] ** 2))
    print(math.sqrt(u_x ** 2 + u_y ** 2))
    DV = []
    for i in range(len(V_x)):
        DV.append(math.sqrt((V_x_new[i] - coords_plane2[i][0]) ** 2 + (V_y_new[i] - coords_plane2[i][1]) ** 2))
    print('max{DV(x, y)} = ', max(DV))
    #масштабирование картинки
    len_coords_0 = math.sqrt((V_x_new[0] - coords_plane2[0][0]) ** 2 + (V_y_new[0] - coords_plane2[0][1]) ** 2)
    len_coords_1 = math.sqrt((V_x_new[m0-1] - coords_plane2[m0-1][0]) ** 2 + (V_y_new[m0-1] - coords_plane2[m0-1][1]) ** 2)
    len_coords_2 = math.sqrt((V_x_new[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][0]) ** 2 + (V_y_new[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][1]) ** 2)
    len_coords_3 = math.sqrt((V_x_new[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][0]) ** 2 + (V_y_new[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][1]) ** 2)
    len_coords = (len_coords_0 + len_coords_1 + len_coords_2 + len_coords_3) / 4
    #print(len_coords)

    for it in range(len(V_x)):
        ax1.quiver(coords_plane2[it][0], coords_plane2[it][1], V_x_new[it] - coords_plane2[it][0], V_y_new[it] - coords_plane2[it][1],  angles = 'xy', scale_units = 'xy', width = 0.0015, scale = len_coords / scale_prm)#scale = len_coords / 0.008)#scale=0.00032 и 0.000042)

    for it in range(len(V_x)):
        l1, l2, l3, l4 = 0, 0, 0, 0
        l1 += math.sqrt((V_x_old[it] - coords_plane2[it][0])**2 + (V_y_old[it] - coords_plane2[it][1])**2)
        l2 += math.sqrt((V_x_alpha[it] - coords_plane2[it][0])**2 + (V_y_alpha[it] - coords_plane2[it][1])**2)
        l3 += math.sqrt((V_x_new_no_alpha[it] - coords_plane2[it][0])**2 + (V_y_new_no_alpha[it] - coords_plane2[it][1])**2)
        l4 += math.sqrt((V_x_new[it] - coords_plane2[it][0])**2 + (V_y_new[it] - coords_plane2[it][1])**2)
    print("Сумма длин векторов поля смещений с учетом вычета вектора центра ПЗС-линейки:", l1)
    print("Сумма длин векторов поля смещений с учетом вычета вектора центра ПЗС-линейки и оптимального угла вращения:", l2)
    print("Сумма длин векторов поля смещений с учетом вычета оптимального компенсирующего вектора, но без учета угла:", l3)
    print("Сумма длин векторов поля смещений после минимизации:", l4)
    ax1.grid(color = 'green')
    ax1.set_aspect(abs((b0/a0)))
    ax1.set_xlabel('OX, дм')
    ax1.set_ylabel('OY, дм')
    plt.xlim([-(b0 + 6 * m1), b0 + 6 * m1])
    plt.ylim([-(a0 + 3 * n1), a0 + 3 * n1])
    plt.show()
    
def maxDV(V_x, V_y, u_x, u_y, coords_plane2):
    DV = []
    for i in range(len(V_x)):
        DV.append(math.sqrt((V_x[i] - u_x - coords_plane2[i][0]) ** 2 + (V_y[i] - u_y - coords_plane2[i][1]) ** 2))
    return max(DV)