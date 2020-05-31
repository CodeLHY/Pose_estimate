from get_mean_score import *
from get_fvector_mul import *
from video_match import *
import cv2
def get_total_frames(path_video):
    cap = cv2.VideoCapture(path_video)
    total_frames = int(cap.get(7))
    return total_frames


def get_matched_score(matches, start_mixed_standard, fv_mul_front_standard, fpositions_left_user, fpositions_front_user,
                          fpositions_left_standard, fpositions_front_standard):
    """
    获取所有匹配之后的停止帧的得分， 注意是以标准视频为依据，就是说：标准视频的每一个停止帧都会有对应的用户得分
    :param matches:
    :param start_mixed_standard: 双视角混合之后的标准视频的起始帧
    :param fv_mul_front_standard: 标准视频前方摄像头的帧间向量积
    :param fpositions_left_user: 用户的左边摄像头拍摄到针个视频的各个帧中所有关节的位置字典
    :param fpositions_front_user: 用户的前方摄像头拍摄到针个视频的各个帧中所有关节的位置字典
    :param fpositions_left_standard: 标准视频的左边摄像头拍摄到针个视频的各个帧中所有关节的位置字典
    :param fpositions_front_standard: 标准视频的前方摄像头拍摄到针个视频的各个帧中所有关节的位置字典
    :return: 一个dict{标准视频的某个动作的停止帧数：用户在这段动作内的得分（起始就是对应的停止帧的得分）}
    :list_not_matches      : 一个list，标准视频中找不到对应帧的序号,比如在这里是[1170,1217]
    """
    # list_stop_standard = []
    dict_stop_score = {}
    list_not_matches=[]
    for k in range(len(matches)):
        if matches[k][0] == None:
             all_score = 50
             limbs_score = {i:0 for i in range(11)}
             stop_frame_user=matches[k][1]
             list_not_matches.append(stop_frame_user)
        else:
            start_frame = start_mixed_standard[k]
            stop_frame_user, stop_frame_standard = matches[k]
            limbs_score, all_score = get_all_scores(fpositions_front_standard, fpositions_left_standard, fpositions_front_user, fpositions_left_user, fv_mul_front_standard,
                                                        start_frame, stop_frame_standard, stop_frame_user)
        limbs_score[11] = all_score  # 将二者组合成一个dict
        # list_stop_standard.append(stop_frame_standard)
        dict_stop_score[stop_frame_user] = limbs_score
    return dict_stop_score,list_not_matches

def get_video_score(total_frames, start_mixed_user, stop_mixed_user, dict_score,list_not_matches):
    """
    给整个视频的所有帧都提供一个分，在没有动作的地方给他空，也就是全零。 在一个动作的区间内（起始到结束）得分全为这个动作的结束帧匹配后比较得到的分数
    :param total_frames: 用户视频流的总帧数
    :start_mixed_user:用户的起始帧
    ：stop_mixed_user:用户的结束帧
    ：list_not_matches：一个list，标准视频中找不到对应帧的序号,比如在这里是[1170,1217]，对在用户视频该帧前后三帧输出50分
    :return: list 每一个元素都是一个字典，对应着这个帧的各个关节的得分（0~10）以及总体得分（11）. 因此其长度和视频的总帧数一致
    """
    scores = {}     #这里改成了dict,比较直观，也可改为list
    score_wait = {i:0 for i in range(12)}# 0~10表示人体的11个躯干，11表示整体得分
    index_start_user = 0
    index_stop_user = 0
    stop_frame = stop_mixed_user[index_stop_user]
    start_frame = start_mixed_user[index_start_user]
    for i in range(total_frames):
        if i < start_frame :
            # 如果当前帧数小于下一个动作的起始帧，那么就给他全0
            score = score_wait
            scores[i]=score
        elif i>=start_frame and i <stop_frame:
            # 如果当前帧数在一个动作的区间内，那么就给他实际的分数
            for key in dict_score.keys():
                if stop_frame == key:                         #判断用户的该关键帧是否有分数，即该帧在标准视频中有对应帧
                    score = dict_score[stop_frame]
            scores[i] = score

        elif i >=stop_frame:
            # 如果当前帧数在这个动作之后的，那么就要更新分数
            index_start_user += 1
            index_stop_user += 1
            if index_start_user<len(stop_mixed_user):
                # 说明，后面还有停止帧，也就是说后面后面还有动作
                stop_frame = stop_mixed_user[index_stop_user]
                start_frame = start_mixed_user[index_start_user]
                scores[i] = score
            else:
                # 说明后面没有停止帧了，也就是说后面没有动作了
                score = score_wait
                scores[i]=score
    for i in range(len(list_not_matches)):#对在用户视频该帧前后三帧输出50分
        k=list_not_matches[i]
        scores[k-3]=scores[k-2]=scores[k-1]=scores[k]=scores[k+1]=scores[k+2]=scores[k+3]=dict_score[k]
    return scores
# def get_25_match(start_mixed_standard,stop_mixed_standard,fv_mul_front_standard,
#                  fpositions_left_user, fpositions_front_user,
#                  fpositions_left_standard, fpositions_front_standard):
#     dict_stop_score = {}
#     for k in range(len(stop_mixed_standard)):
#         start_frame = start_mixed_standard[k]
#         stop_frame_standard=stop_mixed_standard[k]
#         max_score=0                     #标准关键帧与用户前后25帧匹配的分数的最大值
#         max_frame=0                     #匹配分数最大值所对应的用户帧
#         max_limbs_score={}              #匹配分数最大值所对应的躯干分数和总分
#         for j in range(-25,26):
#             stop_frame_user=stop_mixed_standard[k]+j
#             limbs_score, all_score = get_all_scores(fpositions_front_standard, fpositions_left_standard,
#                                                 fpositions_front_user, fpositions_left_user, fv_mul_front_standard,
#                                                 start_frame, stop_frame_standard, stop_frame_user)
#             limbs_score[11] = all_score
#             if all_score>max_score:
#                 max_score=all_score
#                 max_limbs_score=limbs_score
#                 max_frame=stop_frame_user
#         dict_stop_score[max_frame] = max_limbs_score
#     return dict_stop_score


def get_key_frame_25(stop_frame_standard, stop_frames_user):
    list_inside = []
    for stop_frame_user in stop_frames_user:
        stop_frame_user_shift = stop_frame_user-25
        if (((stop_frame_user_shift)>(stop_frame_standard-30)) and (stop_frame_user<(stop_frame_standard+30))):
            list_inside.append(stop_frame_user)
    print(stop_frame_standard)
    print(list_inside)
    return list_inside
def get_match_key_frame_25(start_mixed_standard,stop_mixed_standard,
                           stop_mixed_user,
                           fv_mul_front_standard,
                           fpositions_left_user, fpositions_front_user,
                           fpositions_left_standard, fpositions_front_standard):
    dict_stop_score={}
    list_matchs = []
    for (stop_frame_standard, start_frame_standard) in zip(stop_mixed_standard, start_mixed_standard):

        stop_frames_user_25 = get_key_frame_25(stop_frame_standard, stop_mixed_user)
        max_score_all = 0
        max_score_limbs = {}
        if len(stop_frames_user_25)==0:
            dict_stop_score[stop_frame_standard] = None
            match = [None, stop_frame_standard]
            # dict_match[stop_frame_standard] = None
            list_matchs.append(match)
        else:
            for stop_frame_user_25 in stop_frames_user_25:
                score_limbs, score_all = get_all_scores(fpositions_front_standard, fpositions_left_standard,
                                                        fpositions_front_user, fpositions_left_user,
                                                        fv_mul_front_standard,
                                                        start_frame_standard, stop_frame_standard, stop_frame_user_25)
                if score_all>max_score_all:
                    max_score_all = score_all
                    max_score_limbs = score_limbs
                    # dict_match[stop_frame_standard] = stop_frame_user_25
                    match = [stop_frame_user_25, stop_frame_standard]
            list_matchs.append(match)
            score_limbs_all = max_score_limbs
            score_limbs_all[11] = max_score_all
            dict_stop_score[stop_frame_standard] = score_limbs_all
    return dict_stop_score, list_matchs
def debug_save_mixed(save_dir, stop_frames, path_video):
    cap = cv2.VideoCapture(path_video)
    ret_val, image = cap.read()
    count = 0
    count_save = 0
    while ret_val:
        if count > 0:
            if count - 1 in stop_frames:
                name = save_dir + "/matched_" + str(count) + "_user.jpg"
                cv2.imwrite(name, image)
                count_save += 1
        ret_val, image = cap.read()
        count += 1
        cv2.waitKey(1)


def debug_print_score_video(scores, path_video):
    cap = cv2.VideoCapture(path_video)
    ret_val, image = cap.read()
    count = 0
    # print(len(data[3]))
    while ret_val:
        score = scores[count][11]
        cv2.putText(image, str(score), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('fv_mul', image)
        ret_val, image = cap.read()
        count += 1
        # print(position[count])
        cv2.waitKey()
def debug_print_score_image(scores, matches, path_video_user, path_video_standard, save_dir):
    list_image_user = []
    list_image_stantard = []
    cap_user = cv2.VideoCapture(path_video_user)
    ret_val_user, image_user = cap_user.read()
    while ret_val_user:
        list_image_user.append(image_user)
        ret_val_user, image_user = cap_user.read()
        cv2.waitKey(1)

    cap_standard = cv2.VideoCapture(path_video_standard)
    ret_val_standard, image_standard = cap_standard.read()
    while ret_val_standard:
        list_image_stantard.append(image_standard)
        ret_val_standard, image_standard = cap_standard.read()
        cv2.waitKey(1)

    for index, match in enumerate(matches):
        if match[0] == None:
            continue
        name_user = save_dir + "/matched_" + str(index) + "_" + str(match[0]) + "_user.jpg"
        name_standard = save_dir + "/matched_" + str(index) + "_" + str(match[1]) + "_standard.jpg"
        score = scores[match[1]][11]
        cv2.putText(list_image_user[match[0]], str(score), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(name_user, list_image_user[match[0]])
        cv2.imwrite(name_standard, list_image_stantard[match[1]])

# xiayao
# path_video_front_user = "../Key_frame/data/double/data526/xiayao/xiayao_front_user5_60.avi"
# path_video_left_user = "../Key_frame/data/double/data526/xiayao/xiayao_left_user5_60.avi"
# path_json_front_user = "../Key_frame/result/double/json526/xiayao/xiayao_front_user5_60.json"  #
# path_json_left_user = "../Key_frame/result/double/json526/xiayao/xiayao_left_user5_60.json"  #
# xiadun
# path_video_front_user = "../Key_frame/data/double/data526/xiadun/xiadun_front_user5_34.avi"
# path_video_left_user = "../Key_frame/data/double/data526/xiadun/xiadun_left_user5_34.avi"
# path_json_front_user = "../Key_frame/result/double/json526/xiadun/xiadun_front_user5_34.json"  #
# path_json_left_user = "../Key_frame/result/double/json526/xiadun/xiadun_left_user5_34.json"  #
# tizhuan
path_video_front_user = "../Key_frame/data/double/data526/tizhuan/tizhuan_front_user5_63.avi"
path_video_left_user = "../Key_frame/data/double/data526/tizhuan/tizhuan_left_user5_63.avi"
path_json_front_user = "../Key_frame/result/double/json526/tizhuan/tizhuan_front_user5_63.json"  #
path_json_left_user = "../Key_frame/result/double/json526/tizhuan/tizhuan_left_user5_63.json"  #

# xiayao shift 36 frames
# path_video_front_user = "./data/data_shift_1s/xiayao/xiayao_front_user5_60.avi"
# path_video_left_user = "./data/data_shift_1s/xiayao/xiayao_left_user5_60.avi"
# path_json_front_user = "./result/result_shift_1s/xiayao/xiayao_front_user5_60.json"  #
# path_json_left_user = "./result/result_shift_1s/xiayao/xiayao_left_user5_60.json"  #
fpositions_left_user = get_fposition(path_json_left_user)
fpositions_front_user = get_fposition(path_json_front_user)
start_frames_left_user, stop_frames_left_user, fv_mul_added_left_user,fv_mul_left_user = json2keyframes(path_json_left_user)
start_frames_front_user, stop_frames_front_user, fv_mul_added_front_user ,fv_mul_front_user= json2keyframes(path_json_front_user)








# xiayao
# path_video_front_standard = "../Key_frame/data/double/data526/xiayao/xiayao_front_standard5_60.avi"
# path_video_left_standard = "../Key_frame/data/double/data526/xiayao/xiayao_front_standard5_60.avi"
# path_json_front_standard = "../Key_frame/result/double/json526/xiayao/xiayao_front_standard5_60.json"  #
# path_json_left_standard = "../Key_frame/result/double/json526/xiayao/xiayao_front_standard5_60.json"  #
# xiadun
# path_video_front_standard = "../Key_frame/data/double/data526/xiadun/xiadun_front_standard5_34.avi"
# path_video_left_standard = "../Key_frame/data/double/data526/xiadun/xiadun_front_standard5_34.avi"
# path_json_front_standard = "../Key_frame/result/double/json526/xiadun/xiadun_front_standard5_34.json"  #
# path_json_left_standard = "../Key_frame/result/double/json526/xiadun/xiadun_front_standard5_34.json"  #
# tizhuan
path_video_front_standard = "../Key_frame/data/double/data526/tizhuan/tizhuan_front_standard5_63.avi"
path_video_left_standard = "../Key_frame/data/double/data526/tizhuan/tizhuan_front_standard5_63.avi"
path_json_front_standard = "../Key_frame/result/double/json526/tizhuan/tizhuan_front_standard5_63.json"  #
path_json_left_standard = "../Key_frame/result/double/json526/tizhuan/tizhuan_front_standard5_63.json"  #

# xiayao shift 36 frames
# path_video_front_standard = "./data/data_shift_1s/xiayao/xiayao_front_standard5_60.avi"
# path_video_left_standard = "./data/data_shift_1s/xiayao/xiayao_left_standard5_60.avi"
# path_json_front_standard = "./result/result_shift_1s/xiayao/xiayao_front_standard5_60.json"  #
# path_json_left_standard = "./result/result_shift_1s/xiayao/xiayao_left_standard5_60.json"  #
fpositions_left_standard = get_fposition(path_json_left_standard)
fpositions_front_standard = get_fposition(path_json_front_standard)
start_frames_left_standard, stop_frames_left_standard, fv_mul_added_left_standard,fv_mul_left_standard= json2keyframes(path_json_left_standard)
start_frames_front_standard, stop_frames_front_standard, fv_mul_added_front_standard ,fv_mul_front_standard= json2keyframes(path_json_front_standard)


start_mixed_standard, stop_mixed_standard = mix_double_view(path_json_left_standard, path_json_front_standard)
start_mixed_user, stop_mixed_user = mix_double_view(path_json_left_user, path_json_front_user)
start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user = filter_mixed(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user)

# debug
dict_stop_score, list_matchs = get_match_key_frame_25(start_mixed_standard,stop_mixed_standard,
                           stop_mixed_user,
                           fv_mul_front_standard,
                           fpositions_left_user, fpositions_front_user,
                           fpositions_left_standard, fpositions_front_standard)
debug_print_score_image(dict_stop_score, list_matchs, path_video_front_user, path_video_front_standard, "./key_images_match")
# debug_save_match2(list_matchs, path_video_front_user, path_video_front_standard, "./key_images_match")
# debug_save_mixed("./key_images_standard", stop_mixed_standard, path_video_front_standard)
# debug_save_mixed("./key_images_user", stop_mixed_user, path_video_front_user)
# debug_plot_match(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user, list_matchs)


# matches = search_match(stop_mixed_user, stop_mixed_standard, start_mixed_standard)
# dict_scores,list_not_matches=get_matched_score(matches, start_mixed_standard, fv_mul_front_standard, fpositions_left_user, fpositions_front_user,
#                          fpositions_left_standard, fpositions_front_standard)
# print(dict_scores)
# total_frames=get_total_frames(path_video_front_user)
#
# dict_scores_25=get_25_match(start_mixed_standard,stop_mixed_standard,fv_mul_front_standard,fpositions_left_user, fpositions_front_user,
#                           fpositions_left_standard, fpositions_front_standard)
# print(dict_scores_25)
"""
print(total_frames)
print(start_mixed_standard)
print(matches)
print(stop_mixed_standard)
print(start_mixed_user)
print(stop_mixed_user)
print(matches)
print(len(dict_scores))
print(len(matches))

video_score=get_video_score(total_frames, start_mixed_user, stop_mixed_user, dict_scores,list_not_matches)
print(video_score)
"""

