import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 10  # Размер вектора, можно изменить

if rank == 0:
    a = np.arange(1, M + 1, dtype=np.float64)
    print(f"Последовательное скалярное произведение: {np.dot(a, a)}")
else:
    a = None

# Подготовка для Scatterv: размеры блоков и смещения
local_M = M // size
rcounts = [local_M] * size
remainder = M % size
for i in range(remainder):
    rcounts[i] += 1
displs = [0] * size
for i in range(1, size):
    displs[i] = displs[i-1] + rcounts[i-1]

# Локальный буфер
a_part = np.empty(rcounts[rank], dtype=np.float64)

# Распределение вектора
comm.Scatterv([a, rcounts, displs, MPI.DOUBLE], a_part, root=0)

# Локальное вычисление
local_dot = np.dot(a_part, a_part)

# Вариант A: Reduce на процесс 0
global_dot_reduce = np.array(0.0, dtype=np.float64) if rank == 0 else None
comm.Reduce(local_dot, global_dot_reduce, op=MPI.SUM, root=0)
if rank == 0:
    print(f"Параллельное (Reduce): {global_dot_reduce}")

# Вариант B: Allreduce на все процессы
global_dot_allreduce = np.array(0.0, dtype=np.float64)
comm.Allreduce(local_dot, global_dot_allreduce, op=MPI.SUM)
print(f"Процесс {rank}: Параллельное (Allreduce): {global_dot_allreduce}")