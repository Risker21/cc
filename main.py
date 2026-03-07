# C:/Users/32252/Desktop/数据挖掘/main.py

def sum_of_numbers():
    # 获取用户输入
    user_input = input("请输入用逗号分隔的数字: ")

    # 分割输入字符串并转换为数字列表
    numbers = [int(num.strip()) for num in user_input.split(',')]

    # 计算总和
    total = sum(numbers)

    # 输出结果
    print(f"这些数字的和为: {total}")

if __name__ == "__main__":
    sum_of_numbers()