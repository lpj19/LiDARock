import numpy as np
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class panel():
    '''
        最小二乘法求平面方程
        params:
    '''
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        a = 0
        A = np.ones((len(self.x), 3))
        for i in range(0, len(self.x)):
            A[i, 0] = self.x[a]
            A[i, 1] = self.y[a]
            a += 1

        # 创建矩阵b
        b = np.zeros((len(self.x), 1))
        a = 0
        for i in range(0, len(self.x)):
            b[i, 0] = self.z[a]
            a = a + 1

        # 通过X=(AT*A)-1*AT*b直接求解
        A_T = A.T
        A1 = np.dot(A_T, A)
        A2 = np.linalg.inv(A1)
        A3 = np.dot(A2, A_T)
        X = np.dot(A3, b)
        A = X[0, 0]
        B = X[1, 0]
        C = X[2, 0]
        print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f ' % (X[0, 0], X[1, 0], X[2, 0]))
        self.A = A
        self.B = B
        self.C = 1
        self.D = C
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.scatter(self.x, self.y, self.z, c='r', marker='o')
        x_p = np.linspace(-5, 2, 100)
        y_p = np.linspace(-15, 0, 100)
        x_p, y_p = np.meshgrid(x_p, y_p)
        z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
        ax1.plot_wireframe(x_p, y_p, z_p, rstride=10, cstride=10)
        plt.show()
    def ol(self):
        A = self.A
        B = self.B
        C = self.C
        D = self.D  #截距
        print(D)
        return A,B,C,D
    def dis_pt2panel(self,x,y,z):
        '''求点到面的距离'''
        pline = '%.5f' %((abs(self.A*x+self.B*y+self.C*z+self.D))/(np.sqrt(self.A*self.A+self.B*self.B+self.C*self.C)))
        return pline

def scope(allfile,file,A,B,C,D):
    '''
    #选定文件区域
    平面向下平移
    :param allfile: 文件路径
    :param file: 文件名的列表
    :param A:
    :param B:
    :param C:
    :param D:
    :return:文件起始和结束位置
    '''
    lenth = len(file)
    file_seat_start = 0
    file_seat_end = lenth-1
    # 文件结束位置
    while True:
        filename_path = allfile + '/' + file[file_seat_end]
        x, y, z = op(filename_path)
        inner = []
        for i in range(len(x)):
            if A * x[i] + B * y[i] + C * z[i] + D == 0:
                inner.append(0)
            elif A * x[i] + B * y[i] + C * z[i] + D > 0:  # 点在平面下方
                inner.append(-1)
            elif A * x[i] + B * y[i] + C * z[i] + D < 0:  # 点在平面上方
                inner.append(1)
        inner = set(inner)
        if len(inner) == 1 and -1 in inner:
            file_seat_end = int(file_seat_end/2)
            if file_seat_end <=0:
                break
        elif len(inner) >= 2 or len(inner) == 1 and 1 in inner:
            file_seat_end = int(file_seat_end*2)  #文件结束在列表的位置
            if file_seat_end >= lenth:
                file_seat_end = lenth - 1
            break

    # 文件开始位置
    while True:
        filename_path = allfile + '/' + file[file_seat_start]
        x, y, z = op(filename_path)
        inner = []
        for i in range(len(x)):
            if A * x[i] + B * y[i] + C * z[i] + D == 0:
                inner.append(0)
            elif A * x[i] + B * y[i] + C * z[i] + D > 0:  # 点在平面下方
                inner.append(-1)
            elif A * x[i] + B * y[i] + C * z[i] + D < 0:  # 点在平面上方
                inner.append(1)
        inner = set(inner)
        if len(inner) == 1 and 1 in inner:
            file_seat_start = file_seat_start + int(lenth / 10)
            if file_seat_start >= lenth:
                file_seat_start = file_seat_start - int(lenth / 10)
                break
        elif len(inner) >= 2 or len(inner) == 1 and -1 in inner:
            file_seat_start = file_seat_start - int(lenth / 10)
            if file_seat_start <= 0:
                file_seat_start = 0
            break
    if file_seat_start >= file_seat_end:
        TP = file_seat_end
        file_seat_end = file_seat_start
        file_seat_start = TP

    return file_seat_start,file_seat_end

def judge_distance(x,y,z,filename,A,B,C,D):
    #判断平面是否经过点云文件
    inner = []
    for i in range(len(x)):
        if A*x[i] + B*y[i] + C*z[i]  + D == 0:
            inner.append(0)
        elif A*x[i] + B*y[i] + C*z[i] + D >0:  #点在平面下方
            inner.append(-1)
        elif A * x[i] + B * y[i] + C* z[i] + D <0:  #点在平面上方
            inner.append(1)
    inner = set(inner)
    if len(inner) >= 2 :
        return filename
    else:
        return None

def op(filename):
    # 获取点云的X,Y,Z
    x = []
    y = []
    z = []
    with open(filename,'r') as f1:
        for line in f1:
            line = line.strip().split(',')
            x.append(float(line[0]))
            y.append(float(line[1]))
            z.append(float(line[2]))
    return x,y,z

def change_type(type,filename_path,savepath, start_idx, end_idx):
    #改写类别
    folder_name = f"{start_idx}_{end_idx}"
    folder_path = os.path.join(savepath, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for path in filename_path:
        true_path = os.path.join(folder_path, os.path.basename(path))
        if os.path.exists(true_path):  # 文件是否存在
            continue
        else:
            with open(path, 'r') as f1:
                lines = f1.readlines()
            with open(true_path, 'w') as f1:
                for line in lines:
                    line = line.strip().split(',')
                    line[-1] = str(type)  # 替换原类型
                    true_type = ','.join(('%s' % id for id in line))
                    with open(true_path, 'a+') as f1:
                        f1.writelines(true_type + '\n')

def Batholith(list1):
    '''
    澡云岩
    :param list: 岩性类别列表
    :return: 澡云岩的索引
    '''
    list1 = Counter(list1).most_common()
    for i in range(len(list1)):
        if 10 in list1[i]:
            return i

def main(allfile, savepath, A, B, C, D, block=-999):
    """
    :param allfile: 读取文件路径（分块数据）
    :param savepath: 保存文件路径（分块数据）
    :param A: 平面方程X系数
    :param B: 平面方程Y系数
    :param C: 平面方程Z系数
    :param D: 平面方程的常数
    :param block:
    :return: 无返回
    """
    try:
        start = time.time()
        file = []
        panel = [A, B, C, D]
        block_number = 0
        panel_list = []
        through_filename_path = []
        now_thruogh_block = 0
        test_block = 0

        while panel[3] >= 00:
            pc = panel.copy()  # append() 函数添加列表时，是添加列表的「引用地址」而不是添加列表内容
            panel_list.append(pc)
            panel[3] -= 0.015

        print('平面创建完毕')
        for i in range(len(panel_list)):
            through_filename_path.append([])

        for foot, boot, file in os.walk(allfile):
            file.sort(key=lambda l: int(re.findall('\d+', l)[0]))  # 排序从小到大
            for second_filename in file:
                filename_path = allfile + '/' + second_filename  # 文件路径
                x, y, z = op(filename_path)  # 获取x,y,z,坐标
                # 判断当前平面是否贯穿点云块
                print(second_filename)
                for i in range(len(panel_list)):
                    part_filename_path = judge_distance(x, y, z, filename_path,
                                                        panel_list[i][0],
                                                        panel_list[i][1],
                                                        panel_list[i][2],
                                                        panel_list[i][3])
                    if part_filename_path is not None:
                        through_filename_path[i].append(part_filename_path)
                        break

        tp = []
        tfp = []
        for panel_number in through_filename_path:
            # 统计被贯穿块的类别
            if panel_number == []:
                continue
            for tf in panel_number:
                block_number += 1
                tfp.append(tf)
                with open(tf, 'r') as f2:
                    line = f2.readline()
                    line = line.strip().split(',')
                    tp.append(int(line[-1]))  # 统计点云块的类别
            print(tf)
            if len(tp) != 0:
                lenth = len(tp)
                if 10 in tp:
                    Bid = Batholith(tp)  # 澡云岩的索引
                    tp = Counter(tp).most_common()  # 每个类别及类别数量[(类别,次数)]
                    B_count = tp[Bid][1]
                    if B_count / lenth >= 0.25:
                        max = tp[Bid][0]
                        change_type(max, tfp, savepath, tp[0][0], tp[-1][0])  # 保存文件区间命名
                        tp.clear()
                        tfp.clear()
                        block_number = 0
                    else:
                        max = tp[0][0]
                        change_type(max, tfp, savepath, tp[0][0], tp[-1][0])  # 保存文件区间命名
                        tp.clear()
                        tfp.clear()
                        block_number = 0
                else:
                    tp = Counter(tp).most_common()
                    max = tp[0][0]
                    change_type(max, tfp, savepath, tp[0][0], tp[-1][0])  # 保存文件区间命名
                    tp.clear()
                    tfp.clear()
                    block_number = 0
            else:
                pass
            print('一个地层统计改正完毕')
            tp = []
            tfp.clear()
        print("任务完成")
    except KeyboardInterrupt:
        print("\n程序已被用户中断。")
        return

if __name__ == '__main__':
    A, B, C, D = 0.018, 0.014, -1, 0.291  # 平面方程的参数
    try:
        main(r'D:\点云数据\加密点云统计计算\加密点云原始点云数据', r'C:\Users\lpj\.spyder-py3\后处理结果2', A, B, C, D)
    except Exception as e:
        print(f"发生错误：{e}")
