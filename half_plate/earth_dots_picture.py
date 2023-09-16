import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from src import V_x_old, V_y_old, w_z

def earth_dots_picture(earth_dots, fsk_center, fsk_x, FSK, t_2):
    #print(coords_plane2)
    fig = plt.figure('Точки в ФСК', figsize = (5, 7), dpi=170)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.067, top=0.97)
    ax = fig.add_subplot()
    
    l_t = [earth_dots[0][0][0], earth_dots[0][0][1]]
    r_b = [earth_dots[-1][-1][0], earth_dots[-1][-1][1]]
    


    ticks_x = np.linspace(l_t[0], r_b[0], 11)
    ticks_y = np.linspace(l_t[1], r_b[1], 11)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.tick_params(axis='both', which='major', labelsize=6)

    for i in range(len(earth_dots)):
        rect = patch.Rectangle((earth_dots[i][-1][0], earth_dots[i][-1][1]),
        earth_dots[i][0][0] - earth_dots[i][-1][0], earth_dots[i][0][1] - earth_dots[i][-1][1], color = 'gray')
        ax.add_patch(rect)

        for it in range(len(earth_dots[i]) - 1):
            ax.scatter(earth_dots[i][it][0], earth_dots[i][it][1], s = 1, color = 'red')
            
            # ax.scatter(earth_dots[-1][len(earth_dots[-1]) - m0][0], earth_dots[-1][len(earth_dots[-1]) - m0][1], s = 1, color = 'red')
            # ax.scatter(earth_dots[0][m0 - 1][0], earth_dots[0][m0 - 1][1], s = 1, color = 'red')
            # ax.scatter(earth_dots[0][15][0], earth_dots[0][15][1], s = 1, color = 'red')
            # ax.scatter(earth_dots[0][-1][0], earth_dots[0][-1][1], s = 1, color = 'red')
            # ax.scatter(earth_dots[5][0][0], earth_dots[5][0][1], s = 1, color = 'red')



    ax.grid(color = 'black', linewidth=0.1)
    ax.set_aspect('equal')
    ax.set_aspect(1)
    ax.set_xlabel('OX, м')
    ax.set_ylabel('OY, м')

    plt.show()