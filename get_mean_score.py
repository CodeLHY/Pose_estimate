from get_fvector_mul import  *
#根据肩宽和胯宽生成用户合成之后的全身的关节坐标
def choose_view(fposition1, fposition2,frame):
    """

    :param dict_view1:  标准视频中正前方摄像头在某个时刻的关键帧图片识别出的关节坐标字典
    :param dict_view0:  标准视频中侧方摄像头在某个时刻的关键帧图片识别出的关节坐标字典
    :return:  choice_upper, choice_lower, dict_standard_combined .    choice_upper:上半身选择哪个视角（1：正前方，0：侧方）choice_lower：下半身选择哪个视角
               dict_standard_combined: 根据肩部和胯部合成出的标准视频中某一帧的各个关节坐标的dict
    """

    coordinates1= get_coordinate(fposition1, frame)
    coordinates0= get_coordinate(fposition2, frame)
    shoulder1=(np.array(coordinates1[2]) - np.array(coordinates1[5])).tolist()
    shoulder_len1=np.linalg.norm(shoulder1)
    shoulder0 = (np.array(coordinates0[8]) - np.array(coordinates0[11])).tolist()
    shoulder_len0 = np.linalg.norm(shoulder0)
    if shoulder_len1>shoulder_len0:
        choice_upper=1
    else:choice_upper=0   # todo : **********************************************

    hip1 = (np.array(coordinates1[2]) - np.array(coordinates1[5])).tolist()
    hip_len1 = np.linalg.norm(hip1)
    hip0 = (np.array(coordinates0[8]) - np.array(coordinates0[11])).tolist()
    hip_len0 = np.linalg.norm(hip0)
    if hip_len1 > hip_len0:
        choice_lower = 1
    else:
        choice_lower = 0  # todo : **********************************************
    return choice_upper,choice_lower
def combine_views(choice_upper, choice_lower, fposition1, fposition2,frame):
    """

    :param choice_upper: 上半身选择哪个视角，1：正前方，0：侧方
    :param choice_lower: 下半身选择哪个视角，1：正前方，0：侧方
    :param dict_1: 正前方视角的关节坐标
    :param dict_0: 侧方视角的关节坐标
    :return: dict_user_combined 用户的合成之后的全身的关节坐标
    """
    dict_user_combined = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
    if choice_upper==1:
        for key in range(8):
            dict_user_combined[key] = fposition1[key][frame]
    else:
        for key in range(8):
            dict_user_combined[key] = fposition2[key][frame]

    if choice_lower==1:
        for key in range(8,14):
            dict_user_combined[key] = fposition1[key][frame]
    else:
        for key in range(8,14):
            dict_user_combined[key]=fposition2[key][frame]
    return dict_user_combined
# 根据输入关键帧的坐标信息，得到各躯干的向量，并得到相邻躯干间的角度self_angles和与对应标准躯干的夹角compare_angles
def get_coordinate(fposition,frame):
    dict_output={}
    for key in fposition.keys():
        dict_output[key]=fposition[key][frame]
    return dict_output
def get_limbs(dict_all_joints):
    """
    通过每个关节的三维坐标获取每一个躯干的三维向量
    :param dict_all_joints:
    :return:
    """
    dict_all_limbs ={}
    dict_all_limbs[0] = (np.array(dict_all_joints[0]) - np.array(dict_all_joints[1])).tolist()
    dict_all_limbs[1] = (np.array(dict_all_joints[2]) - np.array(dict_all_joints[5])).tolist()
    dict_all_limbs[2] = (np.array(dict_all_joints[2]) - np.array(dict_all_joints[3])).tolist()
    dict_all_limbs[3] = (np.array(dict_all_joints[5]) - np.array(dict_all_joints[6])).tolist()
    dict_all_limbs[4] = (np.array(dict_all_joints[3]) - np.array(dict_all_joints[4])).tolist()
    dict_all_limbs[5] = (np.array(dict_all_joints[6]) - np.array(dict_all_joints[7])).tolist()
    dict_all_limbs[6] = (np.array(dict_all_joints[8]) - np.array(dict_all_joints[11])).tolist()
    dict_all_limbs[7] = (np.array(dict_all_joints[8]) - np.array(dict_all_joints[9])).tolist()
    dict_all_limbs[8] = (np.array(dict_all_joints[11]) - np.array(dict_all_joints[12])).tolist()
    dict_all_limbs[9] = (np.array(dict_all_joints[9]) - np.array(dict_all_joints[10])).tolist()
    dict_all_limbs[10] = (np.array(dict_all_joints[12]) - np.array(dict_all_joints[13])).tolist()
    return dict_all_limbs
def get_angle(array_1,array_2):
    """
    计算两个三维向量之间的夹角
    :param array_1:
    :param array_2:
    :return:
    """
    cos_ = np.dot(array_1, array_2)/(np.linalg.norm(array_1)*np.linalg.norm(array_2))
    if -1<cos_<1:
        angle = np.arccos(cos_)*180/np.pi
    else: angle=0
    return angle

def get_self_angle(dict_all_limbs):
    angles={}
    angles[0]= get_angle(np.array(dict_all_limbs[0]), np.array(dict_all_limbs[1]))
    angles[1] = get_angle(np.array(dict_all_limbs[1]), np.array(dict_all_limbs[2]))
    angles[2] = get_angle(np.array(dict_all_limbs[1]), np.array(dict_all_limbs[3]))
    angles[3] = get_angle(np.array(dict_all_limbs[2]), np.array(dict_all_limbs[4]))
    angles[4] = get_angle(np.array(dict_all_limbs[3]), np.array(dict_all_limbs[5]))
    angles[5] = get_angle(np.array(dict_all_limbs[1]), np.array(dict_all_limbs[6]))
    angles[6] = get_angle(np.array(dict_all_limbs[6]), np.array(dict_all_limbs[7]))
    angles[7] = get_angle(np.array(dict_all_limbs[6]), np.array(dict_all_limbs[8]))
    angles[8] = get_angle(np.array(dict_all_limbs[7]), np.array(dict_all_limbs[9]))
    angles[9] = get_angle(np.array(dict_all_limbs[8]), np.array(dict_all_limbs[10]))
    return angles
def get_compare_angle(dict_user_limbs,dict_standard_limbs):
    compare_angles={}
    for key in dict_user_limbs.keys():
      compare_angles[key] = get_angle(np.array(dict_user_limbs[key]), np.array(dict_standard_limbs[key]))
      #print(key,compare_angles[key])
    return compare_angles
#根据对输入的fv_mul进行截取，取得该关键帧从起始帧到结束帧的fv_mul,并根据这个计算该关键帧的躯干权重limbs_weights和角度权重angle_weights
def get_weight_mean(fv_mul):
    weight_fv_mul = {}
    sum = 0
    for key in fv_mul.keys():
        weight_key = np.mean(np.array(fv_mul[key]))
        weight_fv_mul[key] = weight_key
        sum += weight_key
    for key in fv_mul.keys():
        weight_fv_mul[key] = weight_fv_mul[key]/sum
    return weight_fv_mul
def get_frame_weights(fv_mul,start_frame,stop_frame):
    frame_mul = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: []}
    for joint in range(14):
        for frame in range(start_frame, stop_frame):
            frame_mul[joint].append(fv_mul[joint][frame])

    frame_weights=get_weight_mean(frame_mul)
    return  frame_weights
def get_limbs_weights(fv_mul,start_frame,stop_frame):
    frame_weights=get_frame_weights(fv_mul,start_frame,stop_frame)
    limbs_weights={}
    limbs_weights[0]=frame_weights[0]+frame_weights[1]
    limbs_weights[1]=frame_weights[2]+frame_weights[5]
    limbs_weights[2]=frame_weights[2]+frame_weights[3]
    limbs_weights[3]=frame_weights[5]+frame_weights[6]
    limbs_weights[4]=frame_weights[3]+frame_weights[4]
    limbs_weights[5]=frame_weights[6]+frame_weights[7]
    limbs_weights[6]=frame_weights[8]+frame_weights[11]
    limbs_weights[7]=frame_weights[8]+frame_weights[9]
    limbs_weights[8]=frame_weights[11]+frame_weights[12]
    limbs_weights[9]=frame_weights[9]+frame_weights[10]
    limbs_weights[10]=frame_weights[12]+frame_weights[13]


    sum_weights=0
    for key in limbs_weights.keys():
        sum_weights=sum_weights+limbs_weights[key]
    for k in range(11):
        limbs_weights[k]=limbs_weights[k]/sum_weights
    return limbs_weights
def get_angle_weights(limbs_weights):
    angle_weights={}
    angle_weights[0]=limbs_weights[0]+limbs_weights[1]
    angle_weights[1]=limbs_weights[1]+limbs_weights[2]
    angle_weights[2]=limbs_weights[1]+limbs_weights[3]
    angle_weights[3]=limbs_weights[2]+limbs_weights[4]
    angle_weights[4]=limbs_weights[3]+limbs_weights[5]
    angle_weights[5]=limbs_weights[1]+limbs_weights[6]
    angle_weights[6]=limbs_weights[6]+limbs_weights[7]
    angle_weights[7]=limbs_weights[6]+limbs_weights[8]
    angle_weights[8]=limbs_weights[7]+limbs_weights[9]
    angle_weights[9]=limbs_weights[8]+limbs_weights[10]
    sum_weights = 0
    for key in angle_weights.keys():
        sum_weights=sum_weights+angle_weights[key]
    for key in angle_weights.keys():
        angle_weights[key]=angle_weights[key]/sum_weights
    return angle_weights
def get_angle2limbs_weights(angle_weights):
    angle2limbs_weights={}
    sum1=angle_weights[0]+angle_weights[1]+angle_weights[2]+angle_weights[5]
    angle2limbs_weights[0] =angle_weights[0]/sum1
    angle2limbs_weights[1] = angle_weights[1]/sum1
    angle2limbs_weights[2] = angle_weights[2] / sum1
    angle2limbs_weights[3] = angle_weights[5] / sum1
    sum2=angle_weights[1]+angle_weights[3]
    angle2limbs_weights[4]=angle_weights[1]/sum2
    angle2limbs_weights[5]=angle_weights[3]/sum2
    sum3=angle_weights[2]+angle_weights[4]
    angle2limbs_weights[6] = angle_weights[2] / sum3
    angle2limbs_weights[7] = angle_weights[4] / sum3
    sum6=angle_weights[5]+angle_weights[6]+angle_weights[7]
    angle2limbs_weights[8] = angle_weights[5] / sum6
    angle2limbs_weights[9] = angle_weights[6] / sum6
    angle2limbs_weights[10] = angle_weights[7] / sum6
    sum7=angle_weights[6]+angle_weights[8]
    angle2limbs_weights[11] = angle_weights[6] / sum7
    angle2limbs_weights[12] = angle_weights[8] / sum7
    sum8 = angle_weights[7] + angle_weights[9]
    angle2limbs_weights[13] = angle_weights[7] / sum8
    angle2limbs_weights[14] = angle_weights[9] / sum8
    return angle2limbs_weights
#分别计算相邻躯干夹角之差得到的分数self_score和对应躯干向量夹角得到的分数compare_score,加权相加后得到最后分数
def get_score(angle_user, angle_standard,alpha_1):
    diff = abs(angle_user - angle_standard)
    score = 10*max(10 - alpha_1*diff/18, 0)
    return score
def get_self_score(self_angles_user, self_angles_standard,angle_weighs,alpha_1):
    score=[]
    scoreds={}
    for key in self_angles_user.keys():
        scored= get_score(self_angles_user[key], self_angles_standard[key],alpha_1)
        scoreds[key]=scored
        scored_weight=scored* angle_weighs[key]
        score.append(scored_weight)
    scores=sum(score)
    return scores,scoreds
def get_compare_score(compare_angles,limbs_weighs,alpha_2):
    score = []
    scoreds={}
    for key in compare_angles.keys():
       diff=abs(compare_angles[key])
       scored= 10*max(10 - alpha_2*diff /18, 0)
       scoreds[key]=scored
       scored_weight=scored* limbs_weighs[key]
       score.append(scored_weight)

    scores = sum(score)
    return scores,scoreds
def get_final_score(self_score,compare_score,self_weight,compare_weight):
    score={}
    score=self_weight*self_score+compare_weight*compare_score
    return score
def get_limbs_score(self_all_score,compare_all_score,weights):
    alpha1=0.7
    alpha2=0.3
    angle2limbs_score={}
    limbs_scores=[]
    angle2limbs_score[0]=self_all_score[0]
    angle2limbs_score[1]=self_all_score[0]*weights[0]+self_all_score[1]*weights[1]+self_all_score[2]*weights[2]+self_all_score[5]*weights[3]
    angle2limbs_score[2]=self_all_score[1]*weights[4]+self_all_score[3]*weights[5]
    angle2limbs_score[3]=self_all_score[2]*weights[6]+self_all_score[4]*weights[7]
    angle2limbs_score[4]=self_all_score[3]
    angle2limbs_score[5] = self_all_score[4]
    angle2limbs_score[6] = self_all_score[5]*weights[8]+self_all_score[6]*weights[9]+self_all_score[7]*weights[10]
    angle2limbs_score[7]=self_all_score[6]*weights[11]+self_all_score[8]*weights[12]
    angle2limbs_score[8]=self_all_score[7]*weights[13]+self_all_score[9]*weights[14]
    angle2limbs_score[9]=self_all_score[8]
    angle2limbs_score[10]=self_all_score[9]
    for key in angle2limbs_score.keys():
        limbs_score=alpha1*compare_all_score[key]+alpha2*angle2limbs_score[key]
        limbs_scores.append(limbs_score)
    return limbs_scores
# 以下为测试调参部分代码,输出单帧对比分数
def get_all_scores(fposition1, fposition2,fposition3, fposition4,fv_mul,start_frame,stop_frame1,stop_frame2):
    alpha_1=alpha_2=3
    self_weight=0.7
    compare_weight=0.3
    choice_upper, choice_lower = choose_view(fposition1, fposition2, stop_frame1)
    combine_standard=combine_views(choice_upper, choice_lower,fposition1,fposition2,stop_frame1)
    combine_user=combine_views(choice_upper, choice_lower,fposition3,fposition4,stop_frame2)
    limbs_1=get_limbs(combine_standard)
    limbs_2=get_limbs(combine_user)
    angles_1=get_self_angle(limbs_1)
    angles_2=get_self_angle(limbs_2)
    compare_angles=get_compare_angle(limbs_1,limbs_2)
    limbs_weights=get_limbs_weights(fv_mul,start_frame,stop_frame1)
    angle_weights=get_angle_weights(limbs_weights)
    self_score,self_all_score=get_self_score(angles_1,angles_2,angle_weights,alpha_1)
    compare_score,compare_all_score=get_compare_score(compare_angles,limbs_weights,alpha_2)
    final_score=get_final_score(self_score,compare_score,self_weight,compare_weight)
    return compare_all_score,final_score
#以下部分输出评分分数和各躯干分数，需要接收输入有fposition1, fposition2,fposition3,fposition4,fv_mul,start_frame, stop_frame1,stop_frame2
#其中fposition1, fposition2表示标准的正面和左面视角坐标，fposition3,fposition4表示用户的正面和左面视角坐标


if __name__=="__main__":
    print("aa")