
import mmap

# 打开共享内存文件
shm_file = open('/dev/shm/SharedMemory', 'r+b')

# 创建内存映射对象
shm = mmap.mmap(shm_file.fileno(), 0)

# 读取共享内存中的数据
data = shm.readline().decode().rstrip()

# 输出结果
print(f"Value: {data}")

# 关闭共享内存文件和映射对象
shm.close()
shm_file.close()

