import sys
import matplotlib.pyplot as plt
import numpy as np

def get_speedup_omp(t_mpi, t_omp) -> list:
    sp = [0.0] * len(t_omp)
    for i in range(len(sp)):
        sp[i] = t_mpi[0] / t_omp[i]
    return sp

def get_speedup_mpi(t_mpi) -> list:
    sp = [0.0] * len(t_mpi)
    for i in range(len(sp)):
        sp[i] = t_mpi[0] / t_mpi[i]
    return sp

procs_mpi = [1, 2, 4, 8, 16, 32]
procs_omp = [1, 2, 4, 8]

t_mpi_128 = [10.53250, 6.84805, 3.16360, 1.44000, 1.02310, 0.55020]
t_mpi_256 = [108.45930, 74.6329, 40.80650, 19.11920, 17.01840, 4.53360]
t_mpi_512 = [867.30670, 619.1507, 370.99470, 187.85770, 97.66530, 48.87710]

t_omp_128 = [7.90810, 4.13980, 2.28040, 1.16510]
t_omp_256 = [86.98200, 49.28200, 23.81850, 12.01360]
t_omp_512 = [737.95810, 402.86250, 231.57870, 113.69860]

speedup_mpi_128 = get_speedup_mpi(t_mpi_128)
speedup_mpi_256 = get_speedup_mpi(t_mpi_256)
speedup_mpi_512 = get_speedup_mpi(t_mpi_512)

speedup_omp_128 = get_speedup_omp(t_mpi_128, t_omp_128)
speedup_omp_256 = get_speedup_omp(t_mpi_256, t_omp_256)
speedup_omp_512 = get_speedup_omp(t_mpi_512, t_omp_512)

print(speedup_mpi_128)
print(speedup_mpi_256)
print(speedup_mpi_512)
quit()

plt.plot(procs_mpi, speedup_mpi_512, '-ro', linewidth=2.0)
plt.plot(procs_omp, speedup_omp_512, '-bo', linewidth=2.0)
#plt.plot(procs, e2, '-bo', linewidth=2.0)
#plt.plot(procs, e3, '-go', linewidth=2.0)

plt.title("N = 512")
plt.xlabel("Количество MPI-процессов")
plt.ylabel("Ускорение")

plt.show()