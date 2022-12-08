import sys
import matplotlib.pyplot as plt
import numpy as np

cmd = sys.argv[1]
size = int(sys.argv[2])
if cmd == "an":
    dir = "plot_an"
elif cmd == "ap":
    dir = "plot_ap"
elif cmd == "diff":
    dir = "plot_diff"
else:
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
if cmd == "diff":
    cmap = 'PuBu'
else:
    cmap = 'coolwarm'
sc = ax.scatter(x, y, z, c = points, cmap = cmap)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
print(min(points), max(points))
plt.colorbar(sc, orientation='vertical')
plt.show()