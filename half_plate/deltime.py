import math
from src import R, n_0, w_z

def deltime(fsk_center, fsk_x):
    delta = math.sqrt((fsk_center[0] - fsk_x[0]) ** 2 + (fsk_center[1] - fsk_x[1]) ** 2 + (fsk_center[2] - fsk_x[2]) ** 2)
    t_2 = 2 * math.asin(delta / (R)) / (n_0 - w_z)
    return t_2