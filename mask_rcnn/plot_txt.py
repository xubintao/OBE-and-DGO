import matplotlib.pyplot as plt

# 读取文件并解析每行的第二个值
file_path = "det_results20240412-121505.txt"  # 将文件路径替换为实际文件路径
values = []
with open(file_path, "r") as file:
    for line in file:
        parts = line.split()
        if len(parts) >= 2:
            value = float(parts[2])
            values.append(value)

# 绘制折线图
plt.plot(values)
plt.title('Line Plot of Second Values')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.show()
