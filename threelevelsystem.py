from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import linalg as LA
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

delay = np.linspace(-2 * 1E-7, 2 * 1E-7, 10000)  # time delay s
delay1 = np.linspace(0,2 * 1E-7, 10000)
t = np.linspace(0, 50 * 1E-8, 10000)  # time s
L_3 = [(1.0 / 12.99) * 1E9, (1.0 / 33.33) * 1E9, (1.0 / 303.03) * 1E9]
initils1 = [1.0, 0.0, 0.0]
initils0 = [1.0, 0.0]
power = [50 * 1E5, 50 * 1E6, 10 * 1E7]
k = power[2]

# Differential equation of 3 Level case


def Func1(P, t, k):
    L = [k, L_3]
    P0 = P[0]
    P1 = P[1]
    P2 = P[2]
    P0dot = -L[0] * P0 + L[1][2] * P2 + L[1][0] * P1
    P1dot = L[0] * P0 - L[1][1] * P1 - L[1][0] * P1
    P2dot = L[1][1] * P1 - L[1][2] * P2
    return [P0dot, P1dot, P2dot]


def ans1(k):
    return odeint(Func1, initils1, delay1, (k,))


def norm(k, i):
    return ans1(k)[:, i][-1]


def Func0(P, t, k):
    L = [k, 0.1 * 1E9]
    P0 = P[0]
    P1 = P[1]
    P0dot = -L[0] * P0 + L[1] * P1
    P1dot = L[0] * P0 - L[1] * P1
    return [P0dot, P1dot]


# Differential equation of 2 Level case
def ans0(k):
    return odeint(Func0, initils0, delay1, (k,))


# The equation of g2 2 level case
def G2_2level(delay, k):
    L = (1.0 / 10.0) * 1E9
    return 1 - np.exp(-(k + L) * np.abs(delay))

# The equation of g2 3 level case mine
def G2_3level(delay, k):
    L0 = L_3[0]
    L1 = L_3[1]
    L2 = L_3[2]
    A = k + L0 + L1 + L2
    B = k * L1 + k * L2 + L0 * L2 + L1 * L2
    x = (1.0 / 2.0) * (A + np.sqrt(A ** 2 - 4 * B))
    y = (1.0 / 2.0) * (A - np.sqrt(A ** 2 - 4 * B))
    a = (1 - L2 / x) / (L2 * (1.0 / y - 1.0 / x))
    return (1 - a) * np.exp(-x * np.abs(delay)) + a * np.exp(-y * np.abs(delay))

# The equation of g2 3 level case mathematica
def G2_3Level_thes(delay, k):
    L1, L2, L3 = L_3[0], L_3[1], L_3[2]
    abst = np.abs(delay)
    a = k + L1 + L2 + L3
    b = np.sqrt(a ** 2 - 4 * (k * L2 + (a - L3) * L3))
    c = (L1 + L2 - L3) * L3 + k * (2 * L2 + L3)
    x = k * np.exp((-1.0 / 2.0) * (a + b) * abst) * (c * (np.exp(b * abst) - 1) - b * L3 * (1 + np.exp(b * abst) - 2 * np.exp((1.0 / 2.0) * (a + b) * abst)))
    y = 2 * b * (k * L2 + (a - L3) * L3)
    return x / y


def main1():
    plt.plot(delay1, ans1(power[0])[:, 1] / norm(power[0], 1), color='#ffad33')
    plt.plot(-delay1, ans1(power[0])[:, 1] / norm(power[0], 1), color='#ffad33', label='%g' % power[0])
    plt.plot(delay1, ans1(power[1])[:, 1] / norm(power[1], 1), color='#66a3ff')
    plt.plot(-delay1, ans1(power[1])[:, 1] / norm(power[1], 1), color='#66a3ff', label='%g' % power[1])
    plt.plot(delay1, ans1(power[2])[:, 1] / norm(power[2], 1), color='#ff8080')
    plt.plot(-delay1, ans1(power[2])[:, 1] / norm(power[2], 1), color='#ff8080', label='%g' % power[2])
    plt.xlabel('time delay ~ ns')
    plt.legend(loc='best')
    plt.show()
    return

def main2():
    plt.subplot(2, 1, 1)  # Using differential equation
    plt.plot(delay1, ans1(power[0])[:, 1], color='#ffad33')
    plt.plot(-delay1, ans1(power[0])[:, 1], color='#ffad33', label='%g' % power[0])
    plt.plot(delay1, ans1(power[1])[:, 1], color='#66a3ff')
    plt.plot(-delay1, ans1(power[1])[:, 1], color='#66a3ff', label='%g' % power[1])
    plt.plot(delay1, ans1(power[2])[:, 1], color='#ff8080')
    plt.plot(-delay1, ans1(power[2])[:, 1], color='#ff8080', label='%g' % power[2])
    plt.xlabel('time delay ~ ns')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2) # Using mathematica
    plt.plot(delay, G2_3Level_thes(delay, 10 * 1E7), color='#ff8080')
    plt.plot(delay, G2_3Level_thes(delay, 50 * 1E6), color='#66a3ff')
    plt.plot(delay, G2_3Level_thes(delay, 50 * 1E5), color='#ffad33')
    plt.xlabel('time delay ~ ns')
    plt.legend(loc='best')

    plt.show()
    return

if __name__ == '__main__':
    main1()