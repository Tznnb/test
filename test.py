# import numpy as np
#
# array = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]])
# print(array, end="\n\n")
# print("dimension =", array.ndim)    # 维数（秩）
# print("shape =", array.shape)       # 长度宽度
# print("size =", array.size)         # 总个数
# print("type =", array.dtype)        # 元素类型
# print("occupied bytes =", array.itemsize) # 元素所占字节数
#
# cArray = np.array([1, 2, 3], dtype = np.float64)  # 指定类型



# import numpy as np
# array = np.full((3, 4), 5, dtype=np.int64)
# print(array)



# import numpy as np
# a = np.array([[4, 3],
#               [2, 1]])
# b = np.arange(4).reshape((2, 2))
# print(a, b, sep="\n\n", end="\n\n")
#
# c = a * b
# c_matrix = np.dot(a, b)   # a.dot(b)也可
# print(c, c_matrix, sep="\n\n")


# import numpy as np
# array1 = np.eye(3, dtype = int)     # 3行3列的单位矩阵
# array2 = np.eye(3, 4, dtype = int)  # 3行4列的单位矩阵
# print(array1)
# print()
# print(array2)



# import numpy as np
# array1 = np.random.random((3, 4))
# array2 = np.random.rand(3, 4)
# print(array1, array2, sep="\n\n")


# import numpy as np
# array1 = np.ones((2, 3), dtype=int)
# array2 = np.arange(3)
# print("array1", array1, sep="\n", end="\n\n")
# print("array2", array2, sep="\n", end="\n\n")
#
# print(array1 + array2)
# print()
# print(array1<array2)

# import numpy as np
# array = np.array([np.arange(12).reshape((3, 4)),
#                   np.arange(12, 24).reshape((3, 4))])
# print(array)
#
# print(array[0, 0, 0])
# print(array[0][0][0])
# print(array[:, 0, 0])
# print(array[0, 1, 0:4:2])   # 最后一维表示从0到3且间隔为2
# print(array[0, :, :])       # array[0]也可

# print(array.sum())            # np.sum(array)也可，所有元素求和
# print(np.sum(array, axis=1))  # 数组按1轴方向求和，即按行求和
# print(array.sum(axis=0))      # 数组按0轴方向求和，即按列求和

# import numpy as np
# array = np.arange(14, 2, -1).reshape((3, 4))
# print(array, end="\n\n")
#
# print(np.mean(array))     # 平均数
# print(np.median(array))   # 中位数
# print(np.max(array))      # 最大值
# print(np.min(array))      # 最小值
# print(np.std(array))      # 标准差
# print(np.argmax(array))   # 最大值对应索引
# print(np.argmin(array))   # 最小值对应索引
# print(np.cumsum(array))   # 累积的和
# print(np.cumprod(array))  # 累积的积
# print(np.diff(array))     # 相邻的差
# print(np.nonzero(array))  # 非零数的索引
# print(np.sort(array))     # 排序


# import numpy as np
# array = np.arange(24).reshape(4, 6)
# print(array, end="\n\n")
# print(array.transpose(),end="\n\n")
# print(array.T)

# import numpy as np
# array = np.arange(14, 2, -1).reshape((3, 4))
# print(array, end="\n\n")
# print(np.clip(array, 5, 10), end="\n\n")  # 使array中大于10的变为10，小于5的变为5
# print(array.flatten())    # 使多维变为一维

# import numpy as np
# array = np.arange(3, 15).reshape((3, 4))
# print(array, end="\n\n")
# print(array.flatten(), end="\n\n")
# # array.flat是一个迭代器，可将其改为array.flatten()
# for item in array.flat:
#     print(item)




# print("输出每一行：")
# for row in array:
#     print(row)
# print()
#
# print("输出每一列：")
# for col in array.T:   # 利用转置！！！
#     print(col)

# import numpy as np
# A = np.ones(3, dtype=int)
# B = np.full(3, 2, dtype=int)
# A = A[:, np.newaxis]
# B = B[:, np.newaxis]
# print(A, B, sep="\n", end="\n\n")
# C = np.concatenate((A, B), axis=0)
# D = np.concatenate((A, B), axis=1)
# print(C, D, sep="\n", end="\n\n")

# C = np.hstack((A, B))
# print(C)

# C = np.vstack((A, B))    # vertical stack
# D = np.hstack((A, B))    # horizontal stack
# print(C, D, sep="\n\n")
#
# print(A, A.shape)
# B = A[np.newaxis, :]
# print(B, B.shape)
# C = A[:, np.newaxis]
# print(C, C.shape)




# import numpy as np
# A = np.arange(12).reshape((3, 4))
# print(A, end="\n\n")
# # 不等量分割
# print(np.array_split(A, 3, axis=1), end="\n\n")
# print(np.array_split(A, 2, axis=0), end="\n\n")


# # 第二种方式
# print(np.vsplit(A, 3))
# print(np.hsplit(A, 2))





# import numpy as np
#
# a = np.arange(4)
# b = a.copy()
# a[0] = 5
# print(b is a)
# print(a, b, sep="\n")


# import numpy as np
# data = np.arange(50).reshape(5, 10)
# np.savetxt('data_txt.txt', data, fmt='%d')
# np.savetxt('data_csv.csv', data, fmt='%d', delimiter=',')


# import numpy as np
# data1 = np.loadtxt('data_txt.txt')
# data2 = np.loadtxt('data_csv.csv', dtype=int, delimiter=',')
# print(data1, data2, sep="\n\n")

# class Person2:                 #定义类Person2
#     def __init__(self, name,age): #__init__方法
#         self.name = name     #初始化self.name，即成员变量name（域）
#         self.age = age        #初始化self.age，即成员变量age（域）
#     def say_hi (self):         #定义类Person2的函数sayHi
#         print('您好, 我叫', self.name) #在实例方法中通过self.name读取成员变量name（域）
# p1 = Person2('张三',25)       #创建对象
# p1.say_hi()               #调用对象的方法
# print(p1.age)              #通过p1.age（obj1.变量名）读取成员变量age（域）
# print()
# p1.sex = "male"   # 追加一个实例属性
# print(p1.sex)

# class Person3:
#     count = 0   #定义属性count，表示计数
#     name = "Person"   #定义属性name1，表示名称
#
# print(Person3.count)
# Person3.count += 1
# print(Person3.count, end="\n\n")
#
# p1 = Person3()
# p2 = Person3()
# print((p1.name, p2.name))
# Person3.name = "雇员"     # 即时发生改变
# print((p1.name, p2.name), end="\n\n")
#
# p1.name = "员工" #通过实例对象访问，则属于该实例的实例属性
# print((p1.name, p2.name), end="\n\n")
#
# Person3.addition = "哈哈"    # 在类之外追加一个类属性
# print((p1.addition, p2.addition))

# class A:
#     __name = 'class A'  # 私有类属性
#
#     def __init__(self, my_name):
#         self.__my_name = my_name
#
#     def get_name():
#         print(A.__name)  # 在类方法中访问私有类属性
#
#     def get_my_name(self):
#         print(self.__my_name)

# class Person12:
#     def __init__(self, name):
#         self.__name = name
#     @property
#     def name(self):
#         """I'm the 'x' property."""
#         return self.__name
#     @name.setter
#     def name(self, value):
#         self.__name = value
#     @name.deleter
#     def name(self):
#         del self.__name
#
# p = Person12("tzn")
# print(p.name)
# p.name = "nb"
# print(p.name)
# del p.name
# print(p.name)

# class TemperatureConverter:
#     @staticmethod
#     def c2f(t_c): #摄氏温度到华氏温度的转换
#         t_c = float(t_c)
#         t_f = (t_c * 9/5) + 32
#         return t_f
#     @staticmethod
#     def f2c(t_f): #华氏温度到摄氏温度的转换
#         t_f = float(t_f)
#         t_c = (t_f - 32) * 5 /9
#         return t_c
# #测试代码
# print("1. 从摄氏温度到华氏温度.")
# print("2. 从华氏温度到摄氏温度.")
# choice = int(input("请选择转换方向："))
# if choice == 1:
#     t_c = float(input("请输入摄氏温度： "))
#     t_f = TemperatureConverter.c2f(t_c)
#     print("华氏温度为： {0:.2f}".format(t_f))
# elif choice == 2:
#     t_f = float(input("请输入华氏温度： "))
#     t_c = TemperatureConverter.f2c(t_f)
#     print("摄氏温度为： {0:.2f}".format(t_c))
# else:
#     print("无此选项，只能选择1或2！")

# class Foo:
#     classname = "Foo"
#     def __init__(self, name):
#         self.name = name
#     def f1(self): #实例方法
#         print(self.classname)
#         print(self.name)
#     @staticmethod
#     def f2():     #静态方法
#         print("static")
#     @classmethod
#     def f3(cls): #类方法（不能访问实例属性）
#         print(cls.classname)
# #测试代码
# f = Foo("李")
# f.f1()
# Foo.f2()
# Foo.f3()
#

# class Person3:
#     count = 0  # 定义类域count，表示计数
#
#     def __init__(self, name, age):  # 构造函数
#         self.name = name
#         self.age = age
#         Person3.count += 1  # 创建一个实例时，计数加1
#
#     #         self.__class__.count += 1
#     def __del__(self):  # 析构函数
#         Person3.count -= 1  # 销毁一个实例时，计数减1
#
#     def say_hi(self):
#         print('您好, 我叫', self.name)
#
#     @classmethod
#     def get_count(cls):  # 创建类方法
#         print('总计数为：', cls.count)
#
#
# print('总计数为：', Person3.count)  # 类名访问
# p31 = Person3('张三', 25)  # 创建对象
# p31.say_hi()  # 调用对象的方法
# Person3.get_count()  # 通过类名访问
# p32 = Person3('李四', 28)  # 创建对象
# p32.say_hi()  # 调用对象的方法
# Person3.get_count()  # 通过类名访问
# del p31            #删除对象p31
# Person3.get_count()  #通过类名访问
# del p32            #删除对象p32
# Person3.get_count()  #通过类名访问

#
# class Person:  # 基类
#     def __init__(self, name, age):  # 构造函数
#         self.name = name  # 姓名
#         self.age = age  # 年龄
#
#     def say_hi(self):  # 定义基类方法say_hi
#         print('您好, 我叫{0}, {1}岁'.format(self.name, self.age))
#
#
# class Student(Person):  # 派生类
#     def __init__(self, name, age, stu_id):  # 构造函数
#         #         Person.__init__(self, name, age) #调用基类构造函数
#         super().__init__(name, age)  # 优秀，不用知道继承的父类是什么
#         self.stu_id = stu_id  # 学号
#
#     def say_hi(self):  # 定义派生类方法say_hi
#         #         Person.say_hi(self)    #调用基类方法say_hi
#         super().say_hi()
#         print('我是学生, 我的学号为：', self.stu_id)
#
#
# p1 = Person('张王一', 33)  # 创建对象
# p1.say_hi()
# s1 = Student('李姚二', 20, '2013101001')  # 创建对象
# s1.say_hi()

# # don't delete this line!
# from random import *
# def smart_play(m, n):
#     ls = [m]
#     current = m
#     times = n
#
#     while current and times:
#         a = randint(1, 6)
#         b = randint(1, 6)
#         if a+b == 7:
#             current += 4
#         else:
#             current -= 1
#         ls.append(current)
#         times -= 1
#     return ls[-1]
#
# def expected_outcome(m, n, n_play):
#     count = 0
#     for i in range(n_play):
#         count += smart_play(m, n)
#     return count/n_play
#
# seed(1)
# m = 10
# n_play = 500
# result = []
# for n in range(1, 11):
#     outcome = expected_outcome(m, n, n_play)
#     result.append(outcome)
#     print(outcome)
#
# print("The best static is to play {} time(s).".format(result.index(max(result))+1))
# print("The average benefit of playing one time ", end="")
# if result[0]>m:
#     print("> 0.")
# else:
#     print("< 0.")

# def hanoi(n, _from, _aux, _to):
#     if n == 1: print(_from, '->', _to)
#     else:
#         hanoi(n-1, _from=_from, _aux=_to, _to=_aux)
#         hanoi(1, _from=_from, _aux=_aux, _to=_to)
#         hanoi(n-1, _from=_aux, _aux=_from, _to=_to)
# hanoi(4, 'a', 'b', 'c')
# import math
# def hanoi3(n, _from, _aux, _to, times):
#     if n == 1:
#         times[0] += 1
#     else:
#         hanoi3(n-1, _from, _to, _aux, times)
#         hanoi3(1, _from, _aux, _to, times)
#         hanoi3(n-1, _aux, _from, _to, times)
#
# def hanoi4(n, _from, _aux_1, _aux_2, _to, times):
#     if n == 1:
#         times[0] += 1
#     else:
#         tmp = round(abs(math.sqrt(2 * n + 1)))
#         k = n - tmp + 1
#         hanoi4(k, _from, _aux_2, _to, _aux_1, times)
#         hanoi3(n-k, _from, _aux_2, _to, times)
#         hanoi4(k, _aux_1, _from, _aux_2, _to, times)
#
# n = 15
# times = [0]
# hanoi4(n, 'A', 'B', 'C', 'D', times)
# print(times[0])
#
# s = "---"
# s1 = s.replace("-", "+")
# print(s1)

"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 50)
y1 = x * 5 + 1
y2 = x**2

plt.figure(figsize=(10, 6))

# 取值范围
plt.xlim((0, 5))
plt.ylim(0, 20)

# 标签
plt.xlabel("I am x")
plt.ylabel("I am y")

new_ticks = np.linspace(-5, 5, 21)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-5, 3, 10, 16, 20], [r"$really\ bad$", r"$bad$", r"$normal\ \alpha$", r"$good$", r"$really\ good$"])

# gca = 'get current axis'
ax = plt.gca()
# 使右侧和上侧边框消失
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 使下侧边框为x轴，左侧边框为y轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 改变x,y轴位置
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))


l1, = plt.plot(x, y2, label='male', zorder=0)
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='female', zorder=0)
plt.legend(handles=[l1, l2, ], labels=['a', 'b'], loc='upper right')

x0 = 4
y0 = x0**2
plt.scatter(x0, y0, s=50, color='b')
plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)

# method 1
plt.annotate(r"$x**2=%s$"%y0, xy=(x0, y0), xycoords='data',      # x和y的值，text的文本
             xytext=(+30, -30), textcoords='offset points',      # text的放置位置
             fontsize=10, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))  # 字体、箭头曲线

# method 2
plt.text(4.2, 12.5, r"$(This\ is\ an\ annotation)$",
         fontdict={'size':10, 'color':'b'})    # 文本开始位置为(4.2, 12.5)，之后是文本内容与格式

# 对于x轴和y轴的每个label进行更改
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(8)   # 调整字体大小
    label.set_zorder(1)     # zorder类似于图层，与之对应的前面的plot里要加zorder=0
    label.set_bbox(dict(facecolor='white', edgecolor='grey', alpha=0.7))  # alpha表示不透明度


plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np
n = 1024
# 随机正态分布，平均值为0，方差为1，生成个数为n
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
# 生成颜色的值
T = np.arctan2(Y, X)

plt.scatter(X, Y, s=75, c=T, alpha=0.5)

# 设置范围
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))

# 隐藏坐标
plt.xticks(())
plt.yticks(())

plt.show()
"""

# import matplotlib.pyplot as plt
# import numpy as np
# # 一般情况的用法
# plt.scatter(np.arange(5), np.arange(5))
# plt.show()

# class MyList:  # 定义类MyList
#     def __init__(self, *args):  # 构造函数
#         #         self.__mylist = []  #初始化私有属性，空列表
#         #         for arg in args:
#         #             self.__mylist.append(arg)
#         self.__mylist = list(args)
#
#     def __add__(self, n):  # 重载运算符"+"，每个元素增加n
#         for i in range(0, len(self.__mylist)):
#             self.__mylist[i] += n
#
#     #         return self.__mylist
#
#     def __sub__(self, n):  # 重载运算符"-"，每个元素减少n
#         for i in range(0, len(self.__mylist)):
#             self.__mylist[i] -= n
#
#     def __mul__(self, n):  # 重载运算符"*"，每个元素乘以n
#         for i in range(0, len(self.__mylist)):
#             self.__mylist[i] *= n
#
#     def __truediv__(self, n):  # 重载运算符"/"，每个元素除以n
#         for i in range(0, len(self.__mylist)):
#             self.__mylist[i] /= n
#
#     def __len__(self):  # 对应于内置函数len()，返回列表长度
#         return (len(self.__mylist))
#
#     def __repr__(self):  # 对应于内置函数str()，显示列表
#         str1 = ''
#         for i in range(0, len(self.__mylist)):
#             str1 += str(self.__mylist[i]) + ' '
#         return str1
#
#
# #         return str(self.__mylist)
# # 测试代码
# m = MyList(1, 2, 3, 4, 5)  # 创建对象
# m + 2; print(repr(m))   #每个元素加2
# m - 1; print(str(m))   #每个元素减1
# m * 4; print(m)   #每个元素乘4
# m / 2; print(m)   #每个元素除2
# print(len(m))         #列表长度

# x=10
# y=2
# _min = x if x<y else y
# print(_min)

# #2021/4/4
# #实现单精度二进制数转换为十进制数
# def t_change_t(bn):
#    #符号位
#    #阶码位
#    #尾数位
#    Sign_bit=bn[0:1]
#    Order_code_point=bn[1:9]
#    mantissa=bn[9:-1]
#    Order_code_point_t=0
#    #将阶码位变成十进制数判断移动的位数
#    for i in range(len(Order_code_point)):
#       Order_code_point_t=Order_code_point_t+int(Order_code_point[i])*2**(len(Order_code_point)-i-1)
#    displacement=Order_code_point_t-127
#    #偏移量为大于零的数
#    if displacement>=0:
#       BN_int='1'+mantissa[0:displacement]
#       BN_float=mantissa[displacement:-1]
#    #偏移量为小于零的数
#    else:
#       BN_int='0'
#       BN_f=''
#       for j in range((-displacement)-1):
#          BN_f=BN_f+'0'
#       BN_float=BN_f+'1'+mantissa[0:displacement]
#    BN_int_t=0
#    BN_float_t=0
#    for i in range(len(BN_int)):
#       BN_int_t=BN_int_t+int(BN_int[i])*2**(len(BN_int)-i-1)
#    for i in range(len(BN_float)):
#       BN_float_t=BN_float_t+int(BN_float[i])*2**(-(i+1))
#    Answer=BN_int_t+BN_float_t
#    if Sign_bit=='0':
#       Answer=Answer
#    else:
#       Answer=(-Answer)
#    print(Answer)
#    return Answer
# print(t_change_t('11000001101001000000000000000000'))



# import matplotlib.pyplot as plt
# Labels = 'a', 'b', 'c', 'd'
# data = [15, 30, 45, 10]
# Explode = (0, 0.1, 0, 0)
# plt.pie(data, explode=Explode, labels=Labels, autopct=lambda p:f'{p:.2f}%')
# plt.show()


# dic1 = {'a':1, 'b':2}
# s1 = pd.Series(dic1)
#
# dic2 = {'c':3, 'b':2}
# s2 = pd.Series(dic2)
#
# print(s1 + s2)
import numpy as np
import pandas as pd

# dic = {
#     '省份': ['广东', '广东', '江苏', '浙江', '江苏', '浙江'],
#     '城市': ['深圳', '广州', '苏州', '杭州', '南京', '宁波'],
#     'GDP(亿元)': [22286, 21500, 17319, 12556, 11715, 9846],
#     '人口(万)': [1090, 1404, 1065, 919, 827, 788],
# }
# df = pd.DataFrame(dic)
# df = df.set_index('城市') # 重新设定城市名称为行索引
# print(df)


# import numpy as np
# pi_hat = np.array(3.1408, dtype='f8')
# pi_estimation = pi_hat[0]
# print(pi_estimation)

# i = np.arange(1, 10, 3)
# a = i[0]
# print(a)
# print()
#
# def get_terms(i):
#     tmp = i * 2
#     data = np.around((tmp ** 2) / (tmp ** 2 - 1), 4)
#     return data
#
# # print(get_terms(i))
#
# def get_prod(n):
#     array = np.arange(n, dtype=np.float64) + 1
#     tmp = array * 2
#     data = (tmp ** 2) / (tmp ** 2 - 1)
#     result = np.cumprod(data) * 2
#     return result[-1]
#
# # print(get_prod(10000000))
#
# import numpy as np
# import math
# pi = 3.141592653589793
#
# def change(num):
#     integer = math.trunc(num)
#     fraction = np.around(np.array(num, dtype=np.float64) - integer, 15)
#     string = str(fraction)[2:]
#     return string
#
# def get_correctness(pi_hat):
#     if math.trunc(pi) != math.trunc(pi_hat):
#         return 0
#     else:
#         pi_string = change(pi)
#         pi_hat_string = change(pi_hat)
#         count = 0
#         while pi_string[count] == pi_hat_string[count] and count <= 15:
#             count = count + 1
#         return count/15
#
# pi_hat_1 = get_prod(1000)
# pi_hat_2 = get_prod(10000)
# pi_hat_3 = get_prod(10000000)
#
# print(get_correctness(pi_hat_3))
# print(get_correctness(pi_hat_1) * 0.1 + get_correctness(pi_hat_2) * 0.1 + get_correctness(pi_hat_3) * 0.8)


# %timeit for _ in range(1000): True

# import numpy as np
# import math
# import matplotlib.pyplot as plt
# n = [100, 1000, 10000, 100000, 1000000]
# time = [7.2e-6, 12e-6, 47.5e-6, 1.35e-3, 19.6e-3]
# x = []
# y = []
#
# for i in n:
#     x.append(math.log10(i))
# for i in time:
#     y.append(math.log10(i))
#
# print(x, y, sep="\n")
# plt.xlabel("n (log10)")
# plt.ylabel("average time (log10 ms)")
# plt.plot(x, y)
# plt.show()



# def gen():
#     yield 'A'
#     yield 'B'
#     yield 'C'
#
# g = gen()
# print(''.join(g))



# import requests
# r = requests.get('https://www.baidu.com/')
# r.encoding = 'utf-8'              # 方法一：直接转换Response对象的编码格式
# print(r.text)
#
# r = requests.get('https://www.baidu.com/')
# print(r.content.decode('utf-8'))  # 方法二：对二进制内容以utf-8格式解码


# import requests
# r = requests.get('https://10wallpaper.com/wallpaper/1920x1440/1402/tuscany_spring-Seasons_HD_Wallpaper_1920x1440.jpg')
# path = r'C:\Users\Tzn\Downloads\python\pycharm\requests\wallpaper.jpg'
# with open(path, 'wb') as file:
#     file.write(r.content)



# import time
# import requests
# url = 'https://book.douban.com/subject/1255625/comments/?start={}&limit=20&status=P&sort=new_score'
# url_first = 'https://book.douban.com/subject/1255625/comments/?limit=20&status=P&sort=new_score'
# myHeaders = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.62 Safari/537.36'}
# for index, value in enumerate(range(0, 100, 20), start=1):
#     try:
#         if value == 0:
#             r = requests.get(url_first, headers=myHeaders)
#         else:
#             r = requests.get(url.format(value), headers=myHeaders)
#         r.raise_for_status()
#         r.encoding = 'utf-8'
#         path = 'C:/Users/Tzn/Downloads/python/pycharm/requests/reviewing_page_{}.html'.format(index)
#         with open(path, 'w', encoding='utf-8') as file:
#             file.write(r.text)
#         time.sleep(3)       # 抓取一页评论数据后，休眠3秒再抓取下一页
#     except Exception as ex:
#         print("第{}页采集出错，出错原因:{}".format(index, ex))
#
#
#
# ls = ['a', 'b', 'c']
# for index, value in enumerate(ls, start=1):
#     print(f"{index} is {value}")
#
# for index, value in enumerate(range(20, 100, 20)):
#     print(index, value)
#
#
#
#
#
# import requests
# import bs4
#
#
#
#
# myHeaders = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.33'}
#
# code = '''<html>
# <body bgcolor="#eeeeee">
# 	<style>
# 		.css1 { background-color:yellow; color:green;}
# 		.css2 { font-style:italic;}
# 	</style>
# 	<h1 align="center">这里是标题行</h1>
# 	<p name="p1" class="css1 css2">这是第一段</p>
# 	<p name="p2" class="css1 css2">这是第二段</p>
#
# 	<img src="http://www.baidu.com/img/bd_logo1.png" style="width:200px;height:100px"></img>
# 	<a id='link' href="http://baidu.com">点我跳去百度</a>
# </body>
# </html>'''
# try:
#     r = requests.get('https://book.douban.com/subject/1255625/', timeout=30, headers=myHeaders)
#     r.raise_for_status()
#     r.encoding = 'utf-8'
#     print(r.text)
# except Exception as ex:
#     print("wd")
#
#
# # 新建BeautifulSoup对象，赋值给soup变量
# # 第一个参数是网页的文档源码，第二个参数是解析器类型
# soup = bs4.BeautifulSoup(code, 'html.parser')
#
#
# print(soup.find_all('h1'))
# # 输出：[<h1 align="center">这里是标题行</h1>]
#
# print(soup.find_all('h2'))
# # 输出：[]
#
# print(soup.find_all('p'))
# # 输出：[<p class="css1 css2" name="p1">这是第一段</p>,
# #       <p class="css1 css2" name="p2">这是第二段</p>]
#
# print(soup.find_all(['p', 'a']))
# # 输出：[<p class="css1 css2" name="p1">这是第一段</p>,
# #       <p class="css1 css2" name="p2">这是第二段</p>,
# #       <a href="http://baidu.com" id="link">点我跳去百度</a>]
#
# # url = requests.get('https://movie.douban.com/subject/30314848/')
# # url.encoding = 'utf-8'
# # print(url.text)
#
#
#
#
# print(soup.p)          # 输出soup对象中的第一个标签对象：<p class="css1 css2" name="p1">这是第一段</p>
# print(type(soup.p))    # 输出对象类型：<class 'bs4.element.Tag'>
# print(soup.p.name)     # 输出标签类型：p
# print(soup.p.attrs)    # 输出标签属性字典：{'name': 'p1', 'class': ['css1', 'css2']}
# print(soup.p['name'])  # 输出标签name属性：p1
# print(soup.p['class']) # 输出标签class属性：['css1', 'css2']


# import bs4
#
# code = '''<html><head>
# <title>网页标题</title></head>
# <body><h2>金庸群侠传</h2>
# <table width="400px" border="1">
#         <tr><th>书名</th> <th>人物</th> <th>年份</th></tr>
#         <tr><td>《射雕英雄传》</td> <td>郭靖</td> <td>1959年</td></tr>
#         <tr><td>《倚天屠龙记》</td> <td>张无忌</td> <td>1961年</td></tr>
#         <tr><td>《笑傲江湖》</td> <td>令狐冲</td> <td>1967年</td></tr>
#         <tr><td>《鹿鼎记》</td> <td>韦小宝</td> <td>1972年</td></tr>
# </table></body></html>'''
#
# soup = bs4.BeautifulSoup(code, 'html.parser')
# print(soup.find_all('td'))




# score = launchTime = textDocument = usefulness = ['1', '2']
# dict = {'发布时间':launchTime, '评分':score, '有用数':usefulness, '文本':textDocument}
# df = pd.DataFrame(dict, columns=['发布时间', '评分', '有用数', '文本'])
# df

#
import numpy as np
#
# a = np.array([True, True, False, True, False])
#
# print(a.mean())


# import pandas as pd
# dic = {'A':[2, 1, 4, 3], 'B':[9, 10, 8, 7]}
# df = pd.DataFrame(dic)
# print(df)
# res = df.sort_values(by='A', ascending=True)
# print(res)
#
#
# test = {'date':["20220612", "20220611", "20220610", "20220610", "20220610"]}
# pic = pd.DataFrame(test)
# print(pic)
#
# pic['date'] = pd.to_datetime(test['date'], format='%Y%m%d')
# print(pic)
# print(pic['date'])
# pic['delta'] = pd.to_datetime(pic['date'], dayfirst=True)\
#     .sub(pd.to_datetime(pic['date'][0], dayfirst=True))\
#     .dt.days
# pic['score'] = ['5.0', '4.0', '3.0', '3.0', 'NA']
# print(pic)
# ls = []
#
#
# for score in pic['score']:
#     if score != 'NA':
#         ls.append(int(score[0]))
#     else:
#         ls.append(5)
#
# for p, q in zip(pic['delta'], ls):
#     print(p, q)
#
# stastics = {}
# for p, q in zip(pic['delta'], ls):
#     if p in stastics:
#         stastics[p].append(q)
#     else:
#         stastics[p] = [q]
#
# print(stastics)
#
# import numpy as np
# for s in stastics:
#     array = np.array(stastics[s])
#     stastics[s] = array.mean()
# print(stastics)
#
# x = list(stastics.keys())
# y = list(stastics.values())
#
#
# import matplotlib.pyplot as plt
# plt.plot(x, y)
# plt.show()
#
# import numpy as np
# array = np.array([1, 2])
# print(array.size)



# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
# Labels = '正面', '负面', '客观'
# data = [5, 69.1, 25.9]
# plt.pie(data, labels=Labels, autopct='%.2f%%')
# plt.show()


# b = 5
# a = b
# b = 3
# print(a)

# a = [5]
# b = a
# a.append(3)
# print(b)

# week = {'a':'hhh'}
# a = week['a']
# print(a)
# if a == week:
#     print("yes")





print('''abc'ab'abc''')
print('abc\'ab\'abc')

a = str([1, 2])
print(a[0])

