

# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


# confusion = np.array(([63,6,7,1],
#                       [9,90,26,2],
#                       [9,22,90,6],
#                       [13,9,11,47],
#                       ))
# confusion = np.array(([65,5,6,1],
#                       [8,88,27,4],
#                       [9,21,91,6],
#                       [10,7,8,55],
#                       ))
# confusion = np.array(([69,3,5,0],
#                       [6,95,26,0],
#                       [8,21,93,5],
#                       [10,8,6,56],
#                       ))
# confusion = np.array(([62,8,7,0],
#                       [4,99,24,0],
#                       [1,24,97,5],
#                       [8,15,13,44],
#                       ))
# confusion = np.array(([45,15,16,10],
#                       [6,92,4,3],
#                       [2,12,95,3],
#                       [1,3,6,41],
#                       ))
confusion = np.array(([73,2,2,0],
                      [0,119,5,3],
                      [3,0,122,2],
                      [1,3,2,74],
                      ))





classes=['0','1','2','3']
#画出混淆矩阵
def confusion_matrix(confMatrix):
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confMatrix, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confMatrix))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.xticks(indices, classes,rotation=45)
    plt.yticks(indices, classes)

    plt.colorbar()

    # plt.xlabel('预测值')
    # plt.ylabel('真实值')
    # plt.title('混淆矩阵')
    #
    # # plt.rcParams两行是用于解决标签不能显示汉字的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confMatrix)):  # 第几行
        for second_index in range(len(confMatrix[first_index])):  # 第几列
            if first_index==second_index:
                plt.text(first_index, second_index, confMatrix[first_index][second_index],va='center',ha='center',color='white')
            else:
                plt.text(first_index, second_index, confMatrix[first_index][second_index], va='center', ha='center')
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.show()


#计算准确率
def calculate_all_prediction(confMatrix):
    '''
    计算总精度,对角线上所有值除以总数
    :return:
    '''
    total_sum=confMatrix.sum()
    correct_sum=(np.diag(confMatrix)).sum()
    prediction=round(100*float(correct_sum)/float(total_sum),2)
    print('准确率:'+str(prediction)+'%')

def calculae_lable_prediction(confMatrix):
    '''
    计算每一个类别的预测精度:该类被预测正确的数除以该类的总数
    '''
    l=len(confMatrix)
    sum = 0
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=1)[i]
        label_correct_sum=confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        sum+=prediction
        print('精确率:'+classes[i]+":"+str(prediction)+'%')
        P=sum/4
    print('平均精确率:' + ":" + str(P) + '%')
    return P

def calculate_label_recall(confMatrix):
    l = len(confMatrix)
    sum = 0
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        sum += prediction
        print('召回率:'+classes[i] + ":" + str(prediction) + '%')
        R=sum/4
    print('平均召回率:' + ":" + str(R) + '%')
    return R


def calculate_F1():
    P = calculae_lable_prediction(confusion)
    R = calculate_label_recall(confusion)
    F1_score = 2*P*R/(P+R)
    print('F1_score:'  + str(F1_score) + '%')


confusion_matrix(confusion)
calculate_all_prediction(confusion)
calculate_F1()
