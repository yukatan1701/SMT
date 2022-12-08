import sys
import matplotlib.pyplot as plt
import numpy as np

cmd = sys.argv[1]
size = int(sys.argv[2])
approx = True
dir = "plot_ap"
if cmd == "an":
    approx = False
    dir = "plot_an"
elif cmd != "ap":
    print("Invalid mode:", cmd)
    exit(1)

N=size+1
f = open(f"{dir}/{size}.txt", 'r')
points = list(map(float, f.readlines()[0].split()))

x = []
y = []
z = []

for i in range(N):
    for j in range(N):
        for k in range(N):
            x.append(i)
            y.append(j)
            z.append(k)

#f.close()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c = points, cmap = 'coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.colorbar(ax=ax, ticks=[range(0, 100, 10)], orientation='vertical')
plt.show()