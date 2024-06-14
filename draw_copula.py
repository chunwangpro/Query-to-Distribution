import numpy as np
from matplotlib import pyplot as plt

stride = 0.01
X = np.arange(0, 1 + stride, stride)
Y = np.arange(0, 1 + stride, stride)
X, Y = np.meshgrid(X, Y)


def M(x, y, theta=1):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a, b = x[i, j], y[i, j]
            if a <= b:
                z[i, j] = a * b ** (1 - theta)
            else:
                z[i, j] = a ** (1 - theta) * b
    return z


Z = M(X, Y)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)


def P(x, y, theta=0):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a, b = x[i, j], y[i, j]
            if a <= b:
                z[i, j] = a * b ** (1 - theta)
            else:
                z[i, j] = a ** (1 - theta) * b
    return z


Z = P(X, Y)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)


def W(x, y):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = max(x[i, j] + y[i, j] - 1, 0)
    return z


Z = W(X, Y)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)


def f(x, y):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a, b = x[i, j], y[i, j]
            if a + b >= 2 / 3 and a + b <= 4 / 3:
                z[i, j] = min(a, b, 1 / 3, a + b - 2 / 3)
            else:
                z[i, j] = max(a + b - 1, 0)
    return z


Z = f(X, Y)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)

a, b = 0.2, 0.4
Z = a * M(X, Y) + (1 - a - b) * P(X, Y) + b * W(X, Y)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)


def T(x, y, t):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a, b = x[i, j], y[i, j]
            if a >= t and b >= t:
                z[i, j] = max(a + b - 1, t)
            else:
                z[i, j] = min(a, b)
    return z


t = 0.4
Z = T(X, Y, t=t)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z, cmap="coolwarm")
ax1.set_xlabel("u = F(x)")
ax1.set_ylabel("v = G(y)")
ax1.set_zlabel("z")
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap="coolwarm")
plt.show(block=False)

plt.show()
