import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# ======================获得提取关健帧的数据依据的代码===================
def get_fposition(path_json):
    """

    :param path_json: 保存视频中所有帧的人体所有关键点的json 数据,格式：[{0:(x,y), 1:(x,y)...} , {0:(x,y), 1:(x,y)...} ...]
    :return: 经过基础滤波（如果某一帧的某一个关节没有预测值，也就是None那么就给他赋  前一帧/3+后一帧/2 或者 前一帧*2/3+后一帧/3
            或者直接前一帧）之后的坐标，格式：{0：[(x,y),(x,y)...], 1:[(x,y),(x,y)...], .....}
    """
    joints_flow = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    with open(path_json) as f:
        dict_frames = json.load(f)
        for index, joint_frame in enumerate(dict_frames):
            # print(joint_frame.keys())
            if len(joint_frame.keys()) == 0:
                for i in range(14):
                    joints_flow[i].append([0, 0])
            else:
                joints_flow[0].append(joint_frame['0'])
                joints_flow[1].append(joint_frame['1'])
                joints_flow[2].append(joint_frame['2'])
                joints_flow[3].append(joint_frame['3'])
                joints_flow[4].append(joint_frame['4'])
                joints_flow[5].append(joint_frame['5'])
                joints_flow[6].append(joint_frame['6'])
                joints_flow[7].append(joint_frame['7'])
                joints_flow[8].append(joint_frame['8'])
                joints_flow[9].append(joint_frame['9'])
                joints_flow[10].append(joint_frame['10'])
                joints_flow[11].append(joint_frame['11'])
                joints_flow[12].append(joint_frame['12'])
                joints_flow[13].append(joint_frame['13'])
    # get rid of the None element
    for joint in range(14):
        for frame in range(len(joints_flow[joint])):
            if joints_flow[joint][frame] is None:
                if frame == 0: # 判断是否是第0帧
                    joints_flow[joint][frame] = [0, 0]
                elif (frame==(len(joints_flow[joint])-1)): # 判断是否是最后一帧
                        joints_flow[joint][frame] = [1024, 1024]
                        joints_flow[joint][frame][0] = (joints_flow[joint][frame - 1][0])
                        joints_flow[joint][frame][1] = (joints_flow[joint][frame - 1][1])
                elif (frame==(len(joints_flow[joint])-2)): # 判断是否是倒数第二帧
                    if not (joints_flow[joint][frame + 1] is None): # 如果是，那么就判断倒数第一帧是否是None
                        joints_flow[joint][frame] = [1024, 1024]
                        joints_flow[joint][frame][0] = 1 / 2 * (joints_flow[joint][frame-1][0]+joints_flow[joint][frame+1][0])
                        joints_flow[joint][frame][1] = 1 / 2 * (joints_flow[joint][frame - 1][1] + joints_flow[joint][frame + 1][1])

                    else: # 如果最后一帧是None，那么就只能取前面的一个值给他了
                        joints_flow[joint][frame] = [1024, 1024]
                        joints_flow[joint][frame][0] = (joints_flow[joint][frame - 1][0])
                        joints_flow[joint][frame][1] = (joints_flow[joint][frame - 1][1])
                else: # 其余的情况就直接判断后面两帧了
                    if not (joints_flow[joint][frame + 1] is None):  # 如果是，那么就判断倒数第二帧是否是None
                        joints_flow[joint][frame] = [1024, 1024]
                        joints_flow[joint][frame][0] = 1 / 2 * (
                                    joints_flow[joint][frame - 1][0] + joints_flow[joint][frame + 1][0])
                        joints_flow[joint][frame][1] = 1 / 2 * (
                                    joints_flow[joint][frame - 1][1] + joints_flow[joint][frame + 1][1])

                    elif not (joints_flow[joint][frame + 2] is None):  #如果倒数第二帧也是None，那么就判断最后一帧是否是None
                            joints_flow[joint][frame] = [1024, 1024]
                            joints_flow[joint][frame][0] = (1/3*joints_flow[joint][frame - 1][0] + 2/3*joints_flow[joint][frame + 2][0])
                            joints_flow[joint][frame][1] = (1/3*joints_flow[joint][frame - 1][1] + 2/3*joints_flow[joint][frame + 2][1])
                    else:  # 如果最后一帧是None，那么就稚嫩取前面的一个值给他了
                        joints_flow[joint][frame] = [1024, 1024]
                        joints_flow[joint][frame][0] = (joints_flow[joint][frame - 1][0])
                        joints_flow[joint][frame][1] = (joints_flow[joint][frame - 1][1])

    return joints_flow
def get_fvector(joints_flow):
    """

    :param joints_flow: 格式：{0：[(x,y),(x,y)...], 1:[(x,y),(x,y)...], .....}
    :return: 每个帧减去后一个帧得到一个帧间矢量，一共得到frames-1个帧间矢量
            格式：{0：[(x,y),(x,y)...], 1:[(x,y),(x,y)...], .....}
    """
    fvectors = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    for joint in range(14):
        for frame in range(1, len(joints_flow[joint])):
            fvector =  np.array(joints_flow[joint][frame]) - np.array(joints_flow[joint][frame-1])
            fvectors[joint].append(fvector)
    return fvectors
def get_fvector_multiply(fvectors):
    """

    :param fvectors: 格式：{0：[(x,y),(x,y)...], 1:[(x,y),(x,y)...], .....}
    :return: 每个帧间矢量和前一个帧间矢量求点积，得到frames-2个帧间矢量积
            格式：{0：[a,b...], 1:[a,b...], .....}
    """
    fvector_mul = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    for joint in range(14):
        for frame in range(1, len(fvectors[joint])):
            # print(frame-1)
            mul = abs(fvectors[joint][frame-1][0])*abs(fvectors[joint][frame][0]) + abs(fvectors[joint][frame-1][1])*abs(fvectors[joint][frame][1])
            if mul>0.0001:
                cos = np.dot(fvectors[joint][frame-1], fvectors[joint][frame])/(np.linalg.norm(fvectors[joint][frame-1]) * np.linalg.norm(fvectors[joint][frame]))
                # print(cos)
            mul = np.dot(fvectors[joint][frame-1], fvectors[joint][frame])
            fvector_mul[joint].append(abs(mul)) # todo:这里直接用了abs，因为发现负数大多数是由于误测
    return fvector_mul
# =============================可视化代码=============================
def plot_lines(fvector_multiply, start_frames, stop_frames):
    x_list = [i for i in range(len(fvector_multiply))]
    y_list = fvector_multiply
    ax = plt.gca()
    ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)


    for frame in start_frames:
        plt.vlines(frame, 0, 50, color="green")  # 竖线
    for frame in stop_frames:
        plt.vlines(frame, 0, 50, color="yellow")  # 竖线
    plt.show()
def plot_lines_2d(fpositions):
    plt.axis([300, 500, 0, 200])
    plt.ion()

    xs = [fpositions[0][0], fpositions[1][0]]
    ys = [fpositions[0][1], fpositions[1][1]]

    for i in range(300):
        # print(i)
        # print(xs)
        # print(ys)
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = fpositions[i+2][0]
        ys[1] = fpositions[i+2][1]
        plt.plot(xs, ys)
        plt.pause(0.1)
        plt.show()
def print_on_video(data, path_video,frames_start, frames_stop):
    cap = cv2.VideoCapture(path_video)
    ret_val, image = cap.read()
    count = 0
    # print(len(data[3]))
    while ret_val:
        # todo: attention, please,  101 需要三个帧才能得出第一个fv_mul，因此这个值应该和视频中的帧延后三帧才对
        if count > 0:
            if count-1 in frames_start:
                # GET RID OF DATA
                # cv2.putText(image, "START" + str((data[count - 2])), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
                cv2.putText(image, "START", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            elif count-1 in frames_stop:
                # cv2.putText(image, "STOP" + str((data[count - 2])), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
                cv2.putText(image, "STOP", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                # cv2.putText(image, "fv_mul"+str((data[count-1])),(40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
                cv2.putText(image, "MOVING", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(image, "BEGIN", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('fv_mul', image)
        ret_val, image = cap.read()
        count += 1
        # print(position[count])
        cv2.waitKey()
def plot_double_ss(start_frames_front, stop_frames_front,start_frames_left, stop_frames_left):
    for frame in start_frames_front:
        plt.vlines(frame, 0, 50, color="green")  # 竖线
    for frame in stop_frames_front:
        plt.vlines(frame, 0, 50, color="yellow")  # 竖线
    for frame in start_frames_left:
        plt.vlines(frame, 50, 100, color="blue")  # 竖线
    for frame in stop_frames_left:
        plt.vlines(frame, 50, 100, color="red")  # 竖线
    plt.show()
def plot_double_ss_compare(start_frames_front_user, stop_frames_front_user,start_frames_left_user, stop_frames_left_user,
                           start_frames_front_standard, stop_frames_front_standard,start_frames_left_standard, stop_frames_left_standard):
    plt.figure()
    for frame in start_frames_front_user:
        plt.vlines(frame, 0, 50, color="green")  # 竖线
    for frame in stop_frames_front_user:
        plt.vlines(frame, 0, 50, color="yellow")  # 竖线
    for frame in start_frames_left_user:
        plt.vlines(frame, 50, 100, color="blue")  # 竖线
    for frame in stop_frames_left_user:
        plt.vlines(frame, 50, 100, color="red")  # 竖线


    for frame in start_frames_front_standard:
        plt.vlines(frame, 100, 150, color="green")  # 竖线
    for frame in stop_frames_front_standard:
        plt.vlines(frame, 100, 150, color="yellow")  # 竖线
    for frame in start_frames_left_standard:
        plt.vlines(frame, 150, 200, color="blue")  # 竖线
    for frame in stop_frames_left_standard:
        plt.vlines(frame, 150, 200, color="red")  # 竖线
    # plt.show()
# =======================根据数据获取关键帧的代码=======================
def fvecter_filter_dim(delta_i, delta_i_nxt, thre_1, thre_2, thre_abs, reciev_abs):
    """

    :param delta_i: 某一帧 的帧间向量的 某一维度的值
    :param delta_i_nxt: 某一帧的下一帧 的帧间向量的 某一维度的值
    :param thre_1: 两帧之间的阈值
    :param thre_2: 三帧之间的阈值
    :param thre_abs: 绝对接收的阈值
    :return: 滤波之后的值， 当前帧是否处于运动之中， 当前是否处于急速运动之中
    """
    # print(delta_i)
    if abs(delta_i)>thre_abs:
        return delta_i, True, True
    if abs(delta_i)>thre_1:
        return delta_i, True, False
    else:
        if abs(delta_i_nxt)>thre_2:
            return delta_i, True, False
        else:
            if reciev_abs:
                return 1, False, False
            else:
                return 0, False, False
def fvector_filter_core(fvec_i, fvec_i_nxt, in_action, reciev_abs, thre_1_stop, thre_2_stop, thre_1_act, thre_2_act, thre_abs):
    if not in_action:  # 如果不处于运动当中，那么只要大于6的运动我们都能接收，如果不大于6，三帧之间大于7即可
        delta_x, in_action_x, reciev_abs_x = fvecter_filter_dim(fvec_i[0], fvec_i_nxt[0], thre_1_stop, thre_2_stop,thre_abs, reciev_abs)
        delta_y, in_action_y, reciev_abs_y = fvecter_filter_dim(fvec_i[1], fvec_i_nxt[1], thre_1_stop, thre_2_stop,thre_abs, reciev_abs)
        in_action = (in_action_x or in_action_y)
        reciev_abs = (reciev_abs_x or reciev_abs_y)
    else:  # 如果处于运动当中，那么只要大于四的运动我们都能接收，如果不大于4，三帧之间大于5即可，也就是要求放宽了
        delta_x, in_action_x, reciev_abs_x = fvecter_filter_dim(fvec_i[0], fvec_i_nxt[0], thre_1_act, thre_2_act, thre_abs, reciev_abs)
        delta_y, in_action_y, reciev_abs_y = fvecter_filter_dim(fvec_i[1], fvec_i_nxt[1], thre_1_act, thre_2_act, thre_abs, reciev_abs)
        in_action = (in_action_x or in_action_y)
        reciev_abs = (reciev_abs_x or reciev_abs_y)

    return [delta_x, delta_y], in_action, reciev_abs
def fvector_filter(fvec_i, fvec_i_nxt, in_action, recieve_abs, thre_1_stop, thre_2_stop, thre_1_act, thre_2_act, thre_abs):
    """

    :param fvec_i: 当前的帧间向量
    :param fvec_i: 下一个帧间向量
    :param in_action: bool类型 是否处于运动过程中
    :param recieve_abs, 前一帧是否处于急速运动的状态
    :return: 滤波之后当前帧与前一帧的帧间向量， 是否处于运动过程中， 当前是否处于急速运动状态
    """

    fvec_i, in_action, reciev_abs = fvector_filter_core(fvec_i, fvec_i_nxt, in_action, recieve_abs, thre_1_stop, thre_2_stop, thre_1_act, thre_2_act,thre_abs)
    return fvec_i, in_action, reciev_abs
def fvector_filter_all(fvectors_joint):
    fvectors_joint_filtered = []
    in_action = False
    recieve_abs = False
    thre_1_stop = 5
    thre_2_stop = 7
    thre_1_act = 4
    thre_2_act = 5
    thre_abs = 7
    for index in range(len(fvectors_joint)):
        if index==len(fvectors_joint)-1:  # 由于最后一帧 会缺失后一帧的数据，因此直接使用就行了
            fvectors_joint_filtered.append(fvectors_joint[index])
        else:  # 中间帧则需要根据前后帧的结果进行滤波
            fvec_i = fvectors_joint[index]
            fvec_i_nxt = fvectors_joint[index + 1]
            fvec_i_filtered, in_action, reciev_abs = fvector_filter(fvec_i, fvec_i_nxt, in_action, recieve_abs, thre_1_stop, thre_2_stop, thre_1_act, thre_2_act, thre_abs)
            fvectors_joint_filtered.append(fvec_i_filtered)

    return fvectors_joint_filtered
def fvector_filter_all_joints(fvectors):
    fvectors_filted = {}
    for joint in fvectors.keys():
        fvectors_filted[joint] = fvector_filter_all(fvectors[joint])
    return fvectors_filted
def get_weight_mean(fv_mul):
    weight_fv_mul = {}
    sum = 0
    for key in fv_mul.keys():
        weight_key = np.mean(np.array(fv_mul[key]))
        weight_fv_mul[key] = weight_key
        sum += weight_key
    for key in weight_fv_mul.keys():
        weight_fv_mul[key] = weight_fv_mul[key]/sum
        # print(weight_fv_mul[key])
    return weight_fv_mul
def mean_pre_k_frame(fv_muls, index, k):
    sum = 0
    for i in range(k):
        sum += fv_muls[index-i]
    mean = sum/k
    return mean
def mean_nxt_k_frame(fv_muls, index, k):
    sum = 0
    for i in range(k):
        sum += fv_muls[index+i]
    mean = sum/k
    return mean
def mean_near_k_frame(fv_muls, index, k):
    sum = 0
    for i in range(-k//2, k//2):
        sum += fv_muls[index + i]
    mean = sum / k
    return mean
def detect_key(fv_muls):
    """

    :param fv_muls: 帧减向量积序列
    :return: 是否是起始帧， 是否是停止帧    两个都是False的话就是正常帧
    """
    mean = np.mean(np.array(fv_muls))
    start_frames = []
    stop_frames = []
    for index, fv_mul in enumerate(fv_muls):
        if index > 7 and (index<len(fv_muls)-8):
            # todo: 调节这个超参数： mean_pre_k_frame(fv_muls, index, 7)*3  起始帧的判断依据
            if fv_mul<(mean_near_k_frame(fv_muls, index, 10)) or fv_mul==0: # 只要他比周围的10个的平均值-总体的平均值都小，那么就算数
                if mean_pre_k_frame(fv_muls, index, 7)*3<mean_nxt_k_frame(fv_muls, index, 7) and mean_nxt_k_frame(fv_muls, index, 7)*4>mean:
                    start_frames.append(index)
                elif mean_pre_k_frame(fv_muls, index, 7)>mean_nxt_k_frame(fv_muls, index, 7)*3 and mean_pre_k_frame(fv_muls, index, 7)*4>mean:
                    stop_frames.append(index)
    return start_frames, stop_frames
def list_combine(list_1, list_2):
    """
    将两个list组合起来，并且在belone list中标明它属于哪个list，如果维True则属于list1，反之属于list2
    :param list_1:
    :param list_2:
    :return:
    """
    index_1 = 0
    index_2 = 0
    list_combined = []
    list_combined_belong = []
    while index_1<len(list_1) and index_2<len(list_2):
        if list_1[index_1]<=list_2[index_2]:
            list_combined.append(list_1[index_1])
            list_combined_belong.append(True)
            index_1 += 1
        else:
            list_combined.append(list_2[index_2])
            list_combined_belong.append(False)
            index_2 += 1
    if index_1<len(list_1):
        while index_1<len(list_1):
            list_combined.append(list_1[index_1])
            list_combined_belong.append(True)
            index_1 += 1
    elif index_2<len(list_2):
        while index_2<len(list_2):
            list_combined.append(list_2[index_2])
            list_combined_belong.append(False)
            index_2 += 1
    return list_combined, list_combined_belong
def list_divide(list_combined, list_belong):
    list1 = []
    list2 = []
    for index, belong in enumerate(list_belong):
        if belong:
            list1.append(list_combined[index])
        else:
            list2.append(list_combined[index])
    return list1, list2
def del_near_sskey(list_combined, list_belong):
    """
    删掉离得很近的start_frame 和stop_frame
    :param list_combined:
    :param list_belong:
    :return:
    """
    if len(list_belong)<3:
        return list_combined, list_belong
    index_cur = 0
    index_nxt = 1
    belong_cur = list_belong[index_cur]
    belong_nxt = list_belong[index_nxt]
    list_combined_filted = []
    list_belong_filted = []
    while True:
        if belong_cur^belong_nxt:
            if (list_combined[index_nxt]-list_combined[index_cur]) <3:  # 相隔不到两个frame的一定是有误判了，这两个都删掉即可
                index_cur += 2
                index_nxt += 2
                if index_nxt<(len(list_belong)-1):
                    belong_cur = list_belong[index_cur]
                    belong_nxt = list_belong[index_nxt]
                else:
                    break
            else:  # 如果不是，那么就加入到滤波之后得list中
                list_combined_filted.append(list_combined[index_cur])
                list_belong_filted.append(list_belong[index_cur])
                index_cur += 1
                index_nxt += 1
                if index_nxt<=(len(list_belong)-1):
                    belong_cur = list_belong[index_cur]
                    belong_nxt = list_belong[index_nxt]
                else:
                    list_combined_filted.append(list_combined[index_cur])
                    list_belong_filted.append(list_belong[index_cur])
                    break
    # else:  # 如果没有变换性质的话，那就直接加到list上
    #     list_combined_filted.append(list_combined[index_cur])
    #     list_belong_filted.append(list_belong[index_cur])
    #     index_cur += 1
    #     index_nxt += 1
    return list_combined_filted, list_belong_filted
def del_near_same_key(list_combined, list_belong):
    if len(list_belong)<3:
        return list_combined, list_belong
    index_cur = 0
    index_nxt = 1
    length = len(list_combined)  # 记录当前list总长度的变量，防止难以判断是否到达最后一个数据
    belong_cur = list_belong[index_cur]
    belong_nxt = list_belong[index_nxt]
    list_combined_filted = []
    list_belong_filted = []
    flag_stop = False
    while not flag_stop:
        if (belong_cur^belong_nxt):  # 如果相邻的两个是不同属性的，那么久把当前的哪个加入进去
            list_combined_filted.append(list_combined[index_cur])
            list_belong_filted.append(list_belong[index_cur])  # 把当前这个保存起来
            if index_cur<(length-2): # 如果当前不是倒数第二个frame，那么就更新两个belong信息
                index_nxt += 1
                index_cur += 1  # 更新当前和下一个的index
                belong_cur = list_belong[index_cur]
                belong_nxt = list_belong[index_nxt]
            else: # 如果当前已经是倒数第二个frame了，那么已经无法更新belong信息了，那么将最后一个也直接塞进去就好了
                list_combined_filted.append(list_combined[index_nxt])
                list_belong_filted.append(list_belong[index_nxt])
                flag_stop = True # 关闭整个循环
        else:  # 如果当前的和下一个是同一个属性的，那么就取最合理的那个
            temp_index_cur = index_cur
            temp_index_nxt = index_nxt
            temp_frames_same = []
            temp_belong_cur = list_belong[temp_index_cur]
            temp_belong_nxt = list_belong[temp_index_nxt]
            while not temp_belong_cur^temp_belong_nxt:
                temp_frames_same.append(list_combined[temp_index_cur]) # 将当前的这个放入到same list中
                if temp_index_cur<(length-2): #如果这个frame不是倒数第二个
                    temp_index_cur += 1
                    temp_index_nxt += 1  # 更新index
                    temp_belong_cur = list_belong[temp_index_cur]
                    temp_belong_nxt = list_belong[temp_index_nxt]  # 更新belong信息
                    if temp_belong_cur^temp_belong_nxt:
                        temp_frames_same.append(list_combined[temp_index_cur])
                else: # 如果是倒数第二个的话，把最后一个放入到待筛选list中，强制关闭挑选重复帧的循环，并且给总体循环flag置1
                    temp_frames_same.append(list_combined[temp_index_nxt])
                    flag_stop = True
                    break
            # 得到了所有的重复帧，现在来挑选
            if len(list_combined_filted) <3:  # 如果已经被筛选出来的帧数小于3，那就说明前面没有一个参考给我们选择
                mid = len(temp_frames_same)//2
                list_combined_filted.append(temp_frames_same[mid])  # 那么就选取最中间的那个
                list_belong_filted.append(list_belong[index_cur])  # index_cur记载了初次发现重复的时刻，这是的belong信息是我们需要的
                index_cur = temp_index_nxt  # 更新外面的指针到当前的位置
                index_nxt = temp_index_nxt+1
                if not flag_stop:
                    belong_cur = list_belong[index_cur]
                    belong_nxt = list_belong[index_nxt] # 更新从属性质
            else:  # 如果已经被筛选出来的帧数大于等于3： 也就是这种情况：1 0 1 00000 ，则我们可以根据前一个10的距离来选择这次的0
                interval = list_combined_filted[len(list_combined_filted)-2] - list_combined_filted[len(list_combined_filted)-3]
                # 计算所有相同的
                # print(temp_frames_same)
                intervals_temp = [abs(temp_frame - list_combined_filted[len(list_combined_filted)-1] -interval) for temp_frame in temp_frames_same]
                index_min = np.argmin(np.array(intervals_temp))
                list_combined_filted.append(temp_frames_same[index_min])  # 选出最合适的那个
                list_belong_filted.append(list_belong[index_cur])  # index_cur记载了初次发现重复的时刻，这是的belong信息是我们需要的
                index_cur = temp_index_nxt  # 更新外面的指针到当前的位置
                index_nxt = temp_index_nxt + 1
                if not flag_stop:
                    belong_cur = list_belong[index_cur]
                    belong_nxt = list_belong[index_nxt] #更新从属性质

    return list_combined_filted, list_belong_filted
def del_first_stop(list_combined, list_belong):
    if len(list_combined)<3:
        return list_combined, list_belong
    if not list_belong[0]:
        list_combined = list_combined[1::]  # 如果第一个特殊帧是stop帧的话，就把它删掉
        list_belong = list_belong[1::]
    return list_combined, list_belong
def get_closer_sskey(list_combined, list_belong, total_fv_mul):
    if len(list_combined)<2:
        return list_combined, list_belong
    # 首先整个序列的第一个起始帧往前移5帧（max）
    if list_combined[0]>5:
        list_combined[0] = list_combined[0]-5
    elif list_combined[0]>4:
        list_combined[0] = list_combined[0] - 4
    elif list_combined[0]>3:
        list_combined[0] = list_combined[0] - 3
    elif list_combined[0]>2:
        list_combined[0] = list_combined[0] - 2
    elif list_combined[0]>1:
        list_combined[0] = list_combined[0] - 1
    else:
        list_combined[0] = list_combined[0]
    # 1 0 1 0 1 0 1 0 1 0 将第一个1之后的01之间的距离拉近一些，让他们都拉到平静区
    index = 1
    while index <= len(list_combined)-2: # index永远指向停止帧，
        interval = list_combined[index+1] - list_combined[index]
        if interval < 3:
            list_combined[index] += 0
            list_combined[index + 1] -= 0
            index += 2
        else:
            margine = interval//3
            list_combined[index] += margine
            list_combined[index+1] -= margine
            index += 2
    # 如果最后恰好多出来一个停止帧的话，那么单独将他后移7（max）
    if index == (len(list_combined)-1):
        if (list_combined[index]+7)<(total_fv_mul-1):
            list_combined[index] += 7
        elif (list_combined[index]+6)<(total_fv_mul-1):
            list_combined[index] += 6
        elif (list_combined[index]+5)<(total_fv_mul-1):
            list_combined[index] += 5
        elif (list_combined[index]+4)<(total_fv_mul-1):
            list_combined[index] += 4
        elif (list_combined[index]+3)<(total_fv_mul-1):
            list_combined[index] += 3
        elif (list_combined[index]+2)<(total_fv_mul-1):
            list_combined[index] += 2
        elif (list_combined[index]+1)<(total_fv_mul-1):
            list_combined[index] += 1
        elif (list_combined[index]+0)<(total_fv_mul-1):
            list_combined[index] += 0

    return list_combined, list_belong
def filter_key(frames_start, frames_stop, total_fv_mul):
    # print(frames_start)
    # print(frames_stop)
    list_combined_filted,  list_belong_filted = list_combine(frames_start,frames_stop)  # 没问题
    # print(list_combined)
    list_combined_filted, list_belong_filted = del_near_same_key(list_combined_filted, list_belong_filted) # OK
    #print(list_combined_filted)
    list_combined_filted, list_belong_filted = del_near_sskey(list_combined_filted, list_belong_filted)
    #print(list_combined_filted)
    list_combined_filted, list_belong_filted = del_first_stop(list_combined_filted, list_belong_filted)
    #print(list_combined_filted)
    list_combined_filted, list_belong_filted = get_closer_sskey(list_combined_filted, list_belong_filted,total_fv_mul)
    #print(list_combined_filted)
    frames_start_filted, frames_stop_filted = list_divide(list_combined_filted, list_belong_filted)
    #(frames_start_filted)
    return frames_start_filted, frames_stop_filted
# =========================从视频流中获取关键帧的图像数据=========================
def get_key_image(path_video, frames_start, frames_stop):
    """
    # todo: attention: 当前得到的关键帧都是fv_mul的关键帧，实际的图像帧应该是这个帧数+1，如果直接拿fv的关键帧的index来标记视频的话，需要给index+1
                todo: 由于这里我们是使用的视频流的index，因此index-1才是其真正对应的图像
    :param path_video: 视频的地址
    :param frames_start: 起始帧的list
    :param frames_stop: 结束帧的list
    :return: 返回关键帧对应的图片以及他的始终性质
    """
    cap = cv2.VideoCapture(path_video)
    ret_val, image = cap.read()
    count = 0
    list_key_image = []
    list_property = []
    while ret_val:
        if count > 0:
            if count - 1 in frames_start:
                list_key_image.append(image)
                list_property.append("start")
            elif count - 1 in frames_stop:
                list_key_image.append(image)
                list_property.append("stop")
        ret_val, image = cap.read()
        count += 1
        cv2.waitKey(1)
    return list_key_image, list_property
# ===========================将关键帧图像数据保存起来============================
def save_key_images(key_images, key_property, dir_save):
    for index, (image, property_) in enumerate(zip(key_images, key_property)):
        name = os.path.join(dir_save, str(index) + property_ + ".jpg")
        cv2.imwrite(name, image)
# todo: the final function
def json2keyframes(path_json):
    """

    :param path_json:   某个视频的json数据的地址
    :return:   所有的起始帧和结束帧组成的两个list， 以及每一帧的加权求和过的帧间向量积
    """
    fpositions = get_fposition(path_json)
    fvectors = get_fvector(fpositions)
    fvectors_filted = fvector_filter_all_joints(fvectors)
    fv_mul = get_fvector_multiply(fvectors_filted)
    total_fv_mul = len(fv_mul)
    fv_mul_added = np.zeros(len(fv_mul[0]))
    weight_fv = get_weight_mean((fv_mul))
    for key in fv_mul.keys():
        fv_mul_added += np.array(fv_mul[key]) * weight_fv[key]
    start_frames, stop_frames = detect_key(fv_mul_added)
    start_frames, stop_frames = filter_key(start_frames, stop_frames, total_fv_mul)
    return start_frames, stop_frames, fv_mul_added,fv_mul

# ==============================综合两个视角==================================
def get_mean_interval(start_frames, stop_frames):
    """
    找到某个视频流中，前一动作的结束帧和下一动作的起始帧之间的帧数的均值
    :param start_frames:
    :param stop_frames:
    :return:
    """
    list_combined = list_combine(start_frames, stop_frames)
    interval_added = 0
    interval_count = 0
    index = 1
    while index <= len(list_combined) - 2:  # index永远指向停止帧，
        interval = list_combined[index + 1] - list_combined[index]
        interval_added += interval
        interval_count += 1
        index += 2
    interval_mean = interval_added//interval_count
    return interval_mean

def get_peace_areas(start_frames, stop_frames):
    """
    根据起始帧和结束帧组合起来的list， 寻找所有的平静区。 这个平静区的定义是：由于现在所用到的起始帧和结束帧是经过closer函数调节过的，他们二者
    之间的距离实际上是原本距离的1/3，而没有进行closer调节的区间内则很有可能
    :param frames_ss: 起始帧和结束帧组合起来的list
    :return:
    """
    # print("len(start_frames):", len(start_frames))
    # print("len(stop_frames):", len(stop_frames))
    frames_ss, _ = list_combine(start_frames, stop_frames)
    #print("len(frames_ss):", len(frames_ss))
    index = 1
    list_area = []
    while index <= (len(frames_ss) - 2):  # index永远指向停止帧，
        interval_peace = frames_ss[index + 1] - frames_ss[index]
        interval_move = frames_ss[index] - frames_ss[index-1]
        area = [frames_ss[index]-max(int(interval_peace*2), interval_move//2), frames_ss[index + 1]+max(int(interval_peace*2), interval_move//2)]
        list_area.append(area)
        index += 2
        # print("area  get !")
    return list_area

def judge_in_area(position, area):
    """
    判断一个数字与某一个区间的关系
    :param position: 数字
    :param area: 区间
    :return: 0：在区间的左边， 1 在区间的内部， 2 在区间的右边
    """
    if position < area[0]:
        return 0
    elif position > area[1]:
        return 2
    else:
        return 1

def mix_double_view(path_json_left, path_json_front):
    """

    :param path_json_left: 左视角获得的各个关节坐标的json文件
    :param path_json_front: 前方视角获得的各个关节坐标的json文件
    :return: start_frames_output, stop_frames_output 综合两个视角获得的这个视频的所有起始帧和结束帧
    """
    start_frames_left, stop_frames_left, fv_mul_added_left,fv_mul_left = json2keyframes(path_json_left)
    start_frames_front, stop_frames_front, fv_mul_added_front,fv_mul_front = json2keyframes(path_json_front)
    peace_areas_front = get_peace_areas(start_frames_front, stop_frames_front)
    # print("==================", peace_areas_front)
    # 分开讨论起始帧和结束帧
    start_frames_output = []
    index_start_left = 0
    index_start_front = 0
    # 起始帧：
    while ((index_start_front<len(start_frames_front)) and (index_start_left<len(start_frames_left)) and index_start_front<len(peace_areas_front)):
        # 当两个方向的起始帧都还没有讨论完的时候
        if index_start_left == 0:
            # 第一个起始帧是没有与之相对应的平静区的，因此直接取二者的均值
            start_frames_output.append((start_frames_front[index_start_front] + start_frames_left[index_start_left])//2)
            index_start_front += 1
            index_start_left += 1
        else:
            #print(index_start_front-1)
            peace_area_front = peace_areas_front[index_start_front-1]  # 起始帧从index为1的时候才开始有对应的平静区，因此减一
            relative = judge_in_area(start_frames_left[index_start_left], peace_area_front)
            if relative == 0:
                start_frames_output.append(start_frames_left[index_start_left])  # 如果左视角的起始帧在前视角的
                                                            # 某一起始帧的平静区的左边, 那么就说明前视角没有检测到这个起始帧，
                                                            # 于是将这个起始帧加入到输出的起始帧list中
                index_start_left += 1
            elif relative == 1:
                start_frames_output.append((start_frames_left[index_start_left] + start_frames_front[index_start_front])//2)
                index_start_left += 1
                index_start_front += 1
            else:
                start_frames_output.append(start_frames_front[index_start_front])
                index_start_front += 1
    # 如果其中一个还没有讨论完，那么就直接全部添加到尾部即可
    if index_start_front < len(start_frames_front):
        while(index_start_front < len(start_frames_front)):
            start_frames_output.append(start_frames_front[index_start_front])
            index_start_front += 1
    elif index_start_left< len(start_frames_left):
        while(index_start_left< len(start_frames_left)):
            start_frames_output.append(start_frames_left[index_start_left])
            index_start_left += 1

    # print("start_frame has mixed up")
    # 结束帧
    stop_frames_output = []
    index_stop_left = 0
    index_stop_front = 0
    while ((index_stop_front<len(stop_frames_front)) and (index_stop_left<len(stop_frames_left)) and index_stop_front<len(peace_areas_front)):
        # 当两个方向的起始帧都还没有讨论完的时候
        # print("=====aaaaa", len(peace_areas_front))
        peace_area_front = peace_areas_front[index_stop_front]  # 起始帧从index为0的时候便开始有自己的平静区
        relative = judge_in_area(stop_frames_left[index_stop_left], peace_area_front)
        if relative == 0:
            stop_frames_output.append(stop_frames_left[index_stop_left])  # 如果左视角的起始帧在前视角的
                                                        # 某一起始帧的平静区的左边, 那么就说明前视角没有检测到这个起始帧，
                                                        # 于是将这个起始帧加入到输出的起始帧list中
            index_stop_left += 1
        elif relative == 1:
            stop_frames_output.append((stop_frames_left[index_stop_left] + stop_frames_front[index_stop_front])//2)
            index_stop_left += 1
            index_stop_front += 1
        else:
            stop_frames_output.append(stop_frames_front[index_stop_front])
            index_stop_front += 1
    # 如果其中一个还没有讨论完，那么就直接全部添加到尾部即可
    if index_stop_front < len(stop_frames_front):
        while(index_stop_front < len(stop_frames_front)):
            stop_frames_output.append(stop_frames_front[index_stop_front])
            index_stop_front += 1
    elif index_stop_left<len(stop_frames_left):
        while(index_stop_left<len(stop_frames_left)):
            stop_frames_output.append(stop_frames_left[index_stop_left])
            index_stop_left += 1
    return start_frames_output, stop_frames_output

#  =========================用户与标准视频关键帧的匹配===========================
# 在video_evaluate.py文件中

# ===========================================debug==================================================
# path_json_left_user = "./result/double/xiadun/xiadun_left_user4_32.json"  #
# path_video_left_user = "./data/double/xiadun/xiadun_left_user4_32.avi"
# path_json_front_user = "./result/double/xiadun/xiadun_front_user4_32.json"  #
# path_video_front_user = "./data/double/xiadun/xiadun_front_user4_32.json"
# start_frames_left_user, stop_frames_left_user, fv_mul_added_left_user = json2keyframes(path_json_left_user)
# start_frames_front_user, stop_frames_front_user, fv_mul_added_front_user = json2keyframes(path_json_front_user)
#
# path_json_left_standard = "./result/double/xiayao_left_standard4_30.json"  #
# path_video_left_standard = "./data/double/xiayao_left_standard4_30.avi"
# path_json_front_standard = "./result/double/xiayao_front_standard4_30.json"  #
# path_video_front_standard = "./data/double/xiayao_front_standard4_30.avi"
# start_frames_left_standard, stop_frames_left_standard, fv_mul_added_left_standard = json2keyframes(path_json_left_standard)
# start_frames_front_standard, stop_frames_front_standard, fv_mul_added_front_standard = json2keyframes(path_json_front_standard)
#
# start_mixed_user, stop_mixed_user = mix_double_view(path_json_left_user, path_json_front_user)
# #
# plot_double_ss_compare(start_mixed_user, stop_mixed_user,start_mixed_user, stop_mixed_user,
#                            start_frames_front_user, stop_frames_front_user,start_frames_left_user, stop_frames_left_user)

# plot_double_ss_compare(start_frames_front_user, stop_frames_front_user,start_frames_left_user, stop_frames_left_user,
                           # start_frames_front_standard, stop_frames_front_standard,start_frames_left_standard, stop_frames_left_standard)

# plot_double_ss(start_frames_front, stop_frames_front, start_frames_left, stop_frames_left)
# key_images_standard, key_property_standard = get_key_image(path_video_front_standard, start_frames_front_standard, stop_frames_front_standard)
# key_images_user, key_property_user = get_key_image(path_video_front_user, start_frames_front_user, stop_frames_front_user)
# dir_save_standard = "./key_images/standard"
# dir_save_user = "./key_images/user"
# save_key_images(key_images_standard, key_property_standard, dir_save_standard)
# save_key_images(key_images_user, key_property_user, dir_save_user)
# print_on_video(fv_mul_added, path_video, start_frames, stop_frames)
# plot_lines(fv_mul_added, start_frames, stop_frames)
# ===========================================debug==================================================
