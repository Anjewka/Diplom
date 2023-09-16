import numpy as np
import math
import matplotlib.pyplot as plt
from src import V_x_old, V_y_old

def speed_vector_2(a0, b0, n0, m0, n1, m1, l0, V_x, V_y, coords_plane2, scale_prm, width):
    #print(coords_plane2)
    fig = plt.figure('Min field', figsize = (12, 6), dpi=200)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.09, top=0.99)
    ax = fig.add_subplot()

    ticks_x = np.linspace(-b0, b0, m0)
    ticks_y = np.linspace(-a0, a0, n0)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)

    for it in range(len(V_x)):
        V_x_old.append(V_x[it])# - V_x[l0])
        V_y_old.append(V_y[it])# - V_y[l0])
    #построение поля смещения до минимизации с учетом вычета вектора центра прямоугольника
    #масштабирование картинки
    len_coords_0 = math.sqrt((V_x_old[0] - coords_plane2[0][0]) ** 2 + (V_y_old[0] - coords_plane2[0][1]) ** 2)
    len_coords_1 = math.sqrt((V_x_old[m0-1] - coords_plane2[m0-1][0]) ** 2 + (V_y_old[m0-1] - coords_plane2[m0-1][1]) ** 2)
    len_coords_2 = math.sqrt((V_x_old[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][0]) ** 2 + (V_y_old[len(coords_plane2) - m0 + 1] - coords_plane2[len(coords_plane2) - m0 + 1][1]) ** 2)
    len_coords_3 = math.sqrt((V_x_old[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][0]) ** 2 + (V_y_old[len(coords_plane2) - 1] - coords_plane2[len(coords_plane2) - 1][1]) ** 2)
    len_coords = (len_coords_0 + len_coords_1 + len_coords_2 + len_coords_3) / 4
    #print(len_coords)

    for it in range(170):#int(len(V_x)/5)):
        ax.quiver(coords_plane2[it][0], coords_plane2[it][1], V_x_old[it] - coords_plane2[it][0], V_y_old[it] - coords_plane2[it][1],  angles = 'xy', scale_units = 'xy', width = width, scale = len_coords / scale_prm, color = 'red')#scale = len_coords / 0.008)#scale=0.00032 и 0.000042)
    ax.grid(color = 'green')
    ax.set_aspect('equal')
    ax.set_aspect(abs(b0/(2*a0)))
    ax.set_xlabel('OX, дм')
    ax.set_ylabel('OY, дм')
    plt.xlim([-(b0 + 3 * m1), b0 + 3 * m1])
    plt.ylim([-(a0 + 3 * n1), a0 + 3 * n1])
    plt.show()