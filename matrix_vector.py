import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Чтение данных на процессе 0
if rank == 0:
    with open('in.dat', 'r') as f:
        M = int(f.readline().strip())
        N = int(f.readline().strip())
    
    A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
    x = np.loadtxt('xData.dat', dtype=np.float64)
    
    # Последовательная верификация
    b_seq = np.dot(A.T, x)
    print(f"Последовательный результат: {b_seq}")
else:
    M = None
    N = None
    A = None
    x = None

# Разослать M и N всем
M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)

# Подготовка для Scatterv матрицы A (горизонтальные полосы) и вектора x
local_M = M // size
rcounts = [local_M * N] * size  # Для A: количество элементов
remainder = M % size
for i in range(remainder):
    rcounts[i] += N
displs = [0] * size
for i in range(1, size):
    displs[i] = displs[i-1] + rcounts[i-1]

# Для x: те же rcounts, но без *N
rcounts_x = [local_M] * size
for i in range(remainder):
    rcounts_x[i] += 1
displs_x = [0] * size
for i in range(1, size):
    displs_x[i] = displs_x[i-1] + rcounts_x[i-1]

# Локальные буферы
A_part = np.empty(rcounts[rank] // N * N, dtype=np.float64).reshape(rcounts_x[rank], N)
x_part = np.empty(rcounts_x[rank], dtype=np.float64)

# Распределение A и x
if rank == 0:
    A_flat = A.flatten()
else:
    A_flat = None
comm.Scatterv([A_flat, rcounts, displs, MPI.DOUBLE], A_part, root=0)
comm.Scatterv([x, rcounts_x, displs_x, MPI.DOUBLE], x_part, root=0)

# Локальное вычисление: b_temp = A_part.T @ x_part
b_temp = np.dot(A_part.T, x_part)

# Reduce для суммирования b_temp в b на процессе 0
b = np.empty(N, dtype=np.float64) if rank == 0 else None
comm.Reduce([b_temp, N, MPI.DOUBLE], [b, N, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    print(f"Параллельный результат: {b}")