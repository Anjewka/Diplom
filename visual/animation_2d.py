import numpy as np
from src import image, tt
from matplotlib import animation
import matplotlib.pyplot as plt

#Функция построения 2-д графика
def animate_func_1(num, dataSet, x_trace, y_trace, fig_1, ax_1):
    ax_1.clear()  # очищаем фигуру для обновления линии, точки,

    ax_1.imshow(image, extent = [-180, 180, -90, 90]) #добавляем фоновое изображение
    ax_1.set_title(type(image))

    ax_1.scatter(dataSet[0, :num + 1], dataSet[1, :num + 1], color = 'red', s = 1)
    ax_1.scatter(dataSet[0, num], dataSet[1, num], color = 'black', marker = 'o')
    ax_1.scatter(x_trace[:num + 1], y_trace[:num + 1], color = 'purple', s = 1)
    # добавляем постоянную начальную точку

    ax_1.set_xlim([-180, 180])
    ax_1.set_ylim([-90, 90])

    # Добавляем метки
    ax_1.set_title('Trajectory \nTime = ' + str(np.round(tt[num], decimals=2)) + ' sec')
    ax_1.set_xlabel('долгота')
    ax_1.set_ylabel('широта')

#вставляем изображение карты Земли
def map_2d(dataSet, x_trace, y_trace):
    fig_1 = plt.figure('')
    ax_1 = fig_1.add_subplot()
    line_2d = animation.FuncAnimation(fig_1, animate_func_1, fargs=(dataSet, x_trace, y_trace, fig_1, ax_1), interval = 1, frames = len(tt))
    plt.show()