import math
from src import *

def earth_inter(O, dir):
    B_1 = 2 * np.dot(O, dir)
    C_1 = np.dot(O, O) - R * R
    DISCR = B_1 * B_1 - 4 * C_1
    if DISCR >= 0:
        tmp1 = (-B_1 + math.sqrt(DISCR)) / 2
        tmp2 = (-B_1 - math.sqrt(DISCR)) / 2
        return [tmp1, tmp2]
    return [0, 0]