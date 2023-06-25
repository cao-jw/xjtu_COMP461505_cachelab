import numpy as np
import matplotlib.pyplot as plt


def hex_to_decimal(hex_string):
    return int(hex_string, 16)


class Cache:
    def __init__(self, cache_size, block_size, cache_associativity, pf = 1):
        self.cache_size = cache_size
        self.block_size = block_size
        self.cache_associativity = cache_associativity
        self.block_num = self.cache_size // self.block_size
        self.group_num = self.block_num // self.cache_associativity
        self.numpy_array = np.empty(self.block_num, dtype=np.uint32)  # 只存储标识位
        self.numpy_array = self.numpy_array.reshape((self.group_num, self.cache_associativity))
        self.lru_counter = np.zeros((self.group_num, self.cache_associativity), dtype=np.uint32)
        self.dirty_bit = np.zeros((self.group_num, self.cache_associativity), dtype=bool)
        self.time_stamp = 0
        self.miss = 0
        self.hit = 0
        self.write = 0
        self.replace = 0
        self.isused = np.zeros((self.group_num, self.cache_associativity), dtype=bool)
        self.pf = pf

        # 打印基本信息
        if self.pf == 1:
            print("Cache Information:")
            print(f"Cache Size: {self.cache_size} bytes")
            print(f"Block Size: {self.block_size} bytes")
            print(f"Associativity: {self.cache_associativity}")
            print(f"Number of Blocks: {self.block_num}")
            print(f"Number of Groups: {self.group_num}")
            print(f"Shape of Caches: {self.numpy_array.shape}")
            print("")
        
    def process_input(self, input_array):
        access_type = input_array[0]
        address = input_array[1]

        if access_type == 0:  # 读数据
            if self.pf == 1:
                print("load data from address: ", address)
            self.lru_read(address)
                
        elif access_type == 1:  # 写数据
            if self.pf == 1:
                print("store data to address: ", address)
            self.lru_write(address)
                
            
        self.time_stamp = self.time_stamp + 1

    def lru_read(self, address):
        memory_block_index = address // self.block_size
        cache_group_index = memory_block_index % self.group_num
        flag = 0
        ca = self.numpy_array[cache_group_index]
        way = 0
        for i in range(self.cache_associativity):
            if ca[i] == memory_block_index:
                flag = 1
                way = i
                break
        if flag == 1:
            self.hit = self.hit + 1
            if self.pf == 1:
                print(f"hit at: {cache_group_index}, {way}")
                print("cache_group_index: ", cache_group_index, 
                    "way: ", way)
            self.lru_counter[cache_group_index][way] = self.time_stamp
            self.isused[cache_group_index][way] = True
        else:
            self.miss = self.miss + 1
            lru_way = self.replace_block_in_group(cache_group_index)  # 找到最久未使用的块
            self.numpy_array[cache_group_index][lru_way] = memory_block_index
            self.lru_counter[cache_group_index][lru_way] = self.time_stamp
            if self.pf == 1:
                print(f"Cache misses")
            if self.isused[cache_group_index][lru_way]:
                self.replace = self.replace + 1
                if self.pf == 1:
                    print(f"Cache misses, replaced block: {cache_group_index}, {lru_way}")
            if self.dirty_bit[cache_group_index][lru_way]:
                self.write = self.write + 1
                if self.pf == 1:
                    print("write to memory")
                self.dirty_bit[cache_group_index][lru_way] = False

    def lru_write(self, address):
        
        memory_block_index = address // self.block_size
        cache_group_index = memory_block_index % self.group_num
        flag = 0
        ca = self.numpy_array[cache_group_index]
        way = 0
        for i in range(self.cache_associativity):
            if ca[i] == memory_block_index:
                flag = 1
                way = i
                break
        if flag == 1:
            if self.pf == 1:
                print(f"hit at: {cache_group_index}, {way}")
                print("cache_group_index: ", cache_group_index, 
                    "way: ", way)
            self.lru_counter[cache_group_index][way] = self.time_stamp
            self.dirty_bit[cache_group_index][way] = True
            self.isused[cache_group_index][way] = True
        else:
            lru_way = self.replace_block_in_group(cache_group_index)  # 找到最久未使用的块
            self.numpy_array[cache_group_index][lru_way] = memory_block_index
            self.lru_counter[cache_group_index][lru_way] = self.time_stamp
            if self.dirty_bit[cache_group_index][lru_way]:
                self.write = self.write + 1
                if self.pf == 1:
                    print("write to memory")
            self.dirty_bit[cache_group_index][lru_way] = True
            if self.isused[cache_group_index][lru_way]:
                self.replace = self.replace + 1
                if self.pf == 1:
                    print(f"Cache misses, replaced block: {cache_group_index}, {lru_way}")
            if self.pf == 1:
                print("Cache misses, replaced block")
            
            
    def replace_block_in_group(self, cache_group_index):
        lru_counters = self.lru_counter[cache_group_index]
        min_counter = np.min(lru_counters)
        way_to_replace = np.where(lru_counters == min_counter)[0][0]
        return way_to_replace
        
def get_cache_test(cs, bs, ca):
    file_path = "test_code.txt"

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 处理每一行，并将十六进制转换为十进制
    data = []
    for line in lines:
        elements = line.split()
        converted_elements = [int(elements[0]), hex_to_decimal(elements[1])]
        data.append(converted_elements)

    # 将数据列表转换为 NumPy 数组
    codes = np.array(data)
    a = 0
    write  = 0
    times = len(codes)
    cache_size = cs
    block_size = bs
    cache_associativity = ca
    my_cache = Cache(cache_size = cache_size, 
                    block_size = block_size, 
                    cache_associativity = cache_associativity,
                    pf = 0)
    # t = 0    
    for a in range(len(codes)):
        # inc = 0
        code = codes[a]
        if code[0] == 1:
            write = write + 1
        # if t <= a:
        #     inc = int(input("moves: "))
            # t = t + inc       
        a = a + 1
        my_cache.process_input(code)
    count = np.count_nonzero(my_cache.numpy_array)
    my_cache.write = my_cache.write + count
    missing_rate = my_cache.miss/times
    write_rate = my_cache.write/write
    print("missing rate: ", my_cache.miss/times)
    print("write to memory rate: " , my_cache.write/write)
    return missing_rate, write_rate

def step_by_step(cs, bs, ca):
    file_path = "test_code.txt"

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 处理每一行，并将十六进制转换为十进制
    data = []
    for line in lines:
        elements = line.split()
        converted_elements = [int(elements[0]), hex_to_decimal(elements[1])]
        data.append(converted_elements)

    # 将数据列表转换为 NumPy 数组
    codes = np.array(data)
    a = 0
    write  = 0
    times = len(codes)
    cache_sizes = [8*1024, 16*1024, 32*1024, 64*1024]
    cache_size = cache_sizes[cs]
    block_sizes = [16, 32, 64, 128]
    block_size = block_sizes[bs]
    cache_associativities = [1, 2, 4, 8]
    cache_associativity = cache_associativities[ca]
    my_cache = Cache(cache_size = cache_size, 
                    block_size = block_size, 
                    cache_associativity = cache_associativity)
    t = 0    
    for a in range(len(codes)):
        inc = 0
        code = codes[a]
        if code[0] == 1:
            write = write + 1
        if t <= a:
            inc = int(input("moves: "))
            t = t + inc       
        a = a + 1
        my_cache.process_input(code)
    count = np.count_nonzero(my_cache.numpy_array)
    my_cache.write = my_cache.write + count
    missing_rate = my_cache.miss/times
    write_rate = my_cache.write/write
    print("missing rate: ", missing_rate)
    print("write to memory rate: " , write_rate)


def get_result():
    cache_sizes = [8*1024, 16*1024, 32*1024, 64*1024]
    block_sizes = [16, 32, 64, 128]
    cache_associativities = [1, 2, 4, 8]

    # 创建两个自变量的网格
    bls_grid, cas_grid = np.meshgrid(block_sizes, cache_associativities)

    # 创建空列表来存储数据点的坐标和因变量值
    data_points = []
    mr_values = []
    wr_values = []

    # 执行循环并获取数据
    for cache_size in cache_sizes:
        for b, c in zip(bls_grid.flatten(), cas_grid.flatten()):
            mr, wr = get_cache_test(cache_size, b, c)
            data_points.append([cache_size, b, c])
            mr_values.append(mr)
            wr_values.append(wr)

    # 将数据转换为NumPy数组
    data_points = np.array(data_points)
    mr_values = np.array(mr_values)
    wr_values = np.array(wr_values)

    # 创建第一张图，x轴为cache_size，y轴为block_size，z轴为mr
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    sc1 = ax1.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], c=mr_values, cmap='viridis')
    ax1.set_xlabel('cache_size')
    ax1.set_ylabel('block_size')
    ax1.set_zlabel('cache_associativity')
    ax1.set_title('Missing Rate')
    cbar1 = plt.colorbar(sc1)
    cbar1.set_label('Missing Rate')
    plt.savefig('plot_mr.png')
    plt.close(fig1)

    # 创建第二张图，x轴为cache_size，y轴为block_size，z轴为wr
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], c=wr_values, cmap='viridis')
    ax2.set_xlabel('cache_size')
    ax2.set_ylabel('block_size')
    ax2.set_zlabel('cache_associativity')
    ax2.set_title('Writing Rate')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('Writing Rate')
    plt.savefig('plot_wr.png')
    plt.close(fig2)

    data_points = np.array(data_points)
    mr_values = np.array(mr_values)
    wr_values = np.array(wr_values)

    # 将数据点和因变量值合并为一个数组
    point_values = np.column_stack((data_points, mr_values, wr_values))

    # 保存数据到文本文件
    np.savetxt('point_values.txt', point_values, delimiter=',', fmt='%d,%d,%d,%.6f,%.6f')

# main function


# cs = 0
# bs = 0
# ca = 0
# step_by_step(cs, bs, ca)
get_result()


            








