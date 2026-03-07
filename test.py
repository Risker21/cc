'''
实验目标：
1. 理解并应用鸢尾花数据集的基本概念。
2. 利用Python和scikit-learn库进行数据预处理和模型训练。
3. 评估模型性能，包括准确率、召回率、F1值等指标。
4. 可视化数据和模型结果，帮助理解模型行为。
'''
# 导入numpy库并简写为np，用于数值计算和数组操作
import numpy as np
# 从sklearn.datasets模块导入load_iris函数，用于加载经典的鸢尾花数据集
from sklearn.datasets import load_iris
# 调用load_iris()函数加载数据集，返回一个Bunch对象（类似字典的结构）
iris = load_iris()
# 直接查看iris对象内容（调试用，实际运行时可注释掉）
iris

# 从iris对象中提取特征数据，存储在变量X中
# iris.data包含150个样本，每个样本有4个特征：花萼长度、宽度，花瓣长度、宽度
X = iris.data
# 从iris对象中提取目标标签，存储在变量y中
# iris.target包含150个样本的类别标签：0(setosa)、1(versicolor)、2(virginica)
y = iris.target


# 打印数据集的基本信息，帮助了解数据结构
print("数据集形状：", X.shape)  # 输出(150, 4)，表示150个样本，每个样本4个特征
print("类别标签：", np.unique(y))  # 输出[0 1 2]，表示数据集中包含3个不同的类别
print("特征名称：", iris.feature_names)  # 输出每个特征的中文或英文名称
print("目标名称：", iris.target_names)  # 输出每个类别标签对应的花朵名称


# 从sklearn.model_selection模块导入train_test_split函数，用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# 使用train_test_split函数拆分数据集
# 参数说明：
# X, y：要拆分的特征数据和目标标签
# test_size=0.2：测试集占总数据的20%，训练集占80%
# random_state=42：随机种子，保证每次运行得到相同的拆分结果
# shuffle=True：拆分前打乱数据顺序
# stratify=y：按y的类别分布进行分层抽样，保持训练集和测试集的类别比例与原数据集一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# 打印拆分后的训练集和测试集的形状
print("训练集形状：", X_train.shape)  # 输出(120, 4)，表示120个训练样本，每个样本4个特征
print("测试集形状：", X_test.shape)   # 输出(30, 4)，表示30个测试样本，每个样本4个特征


# 从sklearn.svm模块导入SVC类，用于创建支持向量机分类器
from sklearn.svm import SVC

# 创建SVC分类器实例，使用默认参数
# 默认参数包括：kernel='rbf'（径向基核函数）、C=1.0（正则化参数）等
model = SVC()

# 使用训练集数据训练SVM模型
# 训练过程中，模型会学习特征与标签之间的映射关系
model.fit(X_train, y_train)

# 计算模型在训练集上的准确率
# score方法返回正确分类的样本数占总样本数的比例
acc_train = model.score(X_train, y_train)
print("训练集准确率：", acc_train)  # 输出训练集上的分类准确率


# 模型预测阶段

# 使用训练好的模型对测试集进行预测
# predict方法接收测试集特征数据，返回预测的标签
# X_test真实值：", y_test)  # 输出测试集样本的真实标签值
y_pred = model.predict(X_test)
print("y_test真实值：", y_test)
print("y_pred预测值：", y_pred)  # 输出模型对测试集样本的预测标签值

# 计算模型在测试集上的准确率
# 方法：(y_test == y_pred)会生成一个布尔数组，True表示预测正确，False表示预测错误
# sum()函数统计True的个数（即预测正确的样本数）
# len(y_test)获取测试集样本总数
# 两者相除得到准确率
acc_test = sum(y_test == y_pred) / len(y_test)
print("测试集准确率：", acc_test)  # 输出测试集上的分类准确率
