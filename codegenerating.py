import random
# 生成测试代码
test_code = []
for _ in range(100000):
    access_type = random.randint(0, 1)
    address = random.randint(0, 65535)
    test_code.append((access_type, address))

# 将测试代码写入文本文件
with open("test_code.txt", "w") as file:
    for access_type, address in test_code:
        file.write(f"{access_type} {hex(address)}\n")