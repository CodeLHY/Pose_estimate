from get_mean_score import *
from get_fvector_mul import *
from video_match import *
import cv2
import math
def get_total_frames(path_video):
    cap = cv2.VideoCapture(path_video)
    total_frames = int(cap.get(7))
    return total_frames

# 获得用户的停止帧中所有在标准视频的一个停止帧的前后30帧的停止帧的list
def get_key_frame_25(stop_frame_standard, stop_frames_user):
    list_inside = []
    for stop_frame_user in stop_frames_user:
        stop_frame_user_shift = stop_frame_user
        if (((stop_frame_user_shift)>(stop_frame_standard-30)) and (stop_frame_user<(stop_frame_standard+30))):
            list_inside.append(stop_frame_user)
    # print(stop_frame_standard)
    # print(list_inside)
    return list_inside
# 获得所有用户和标准视频的停止帧的匹配结果，以及标准视频中各个停止帧的得分
def get_match_key_frame_25(start_mixed_standard,stop_mixed_standard,
                           start_mixed_user,stop_mixed_user,
                           fv_mul_front_standard,
                           fpositions_left_user, fpositions_front_user,
                           fpositions_left_standard, fpositions_front_standard):
    """

    :param start_mixed_standard:
    :param stop_mixed_standard:
    :param stop_mixed_user:
    :param fv_mul_front_standard:
    :param fpositions_left_user:
    :param fpositions_front_user:
    :param fpositions_left_standard:
    :param fpositions_front_standard:
    :return: list_matchs list【【user，standard】】得到的所有标准视频中停止帧对应的关键帧，
             dict_stop_score list【dict】得到的所有用户视频中被标准视频匹配了的停止帧的得分
    """
    dict_stop_score={}
    list_matchs = []
    for (stop_frame_standard, start_frame_standard) in zip(stop_mixed_standard, start_mixed_standard):

        stop_frames_user_25 = get_key_frame_25(stop_frame_standard, stop_mixed_user)
        max_score_all = 0
        max_score_limbs = {}
        if len(stop_frames_user_25) == 0:
            score_not_match = {i: 0 for i in range(11)}
            score_not_match[11] = 30
            match = [None, stop_frame_standard]
            # dict_stop_score[stop_frame_standard] = score_not_match  对于这种没有被匹配到的关键帧，用户那边就不做记录
            # list_matchs.append(match)
        else:
            for stop_frame_user_25 in stop_frames_user_25:

                start_frame_user_25 = get_start_matchs(stop_frame_user_25, start_mixed_user, stop_mixed_user)
                score_limbs, score_all = get_all_scores(fpositions_front_standard, fpositions_left_standard,
                                                        fpositions_front_user, fpositions_left_user,
                                                        fv_mul_front_standard,
                                                        start_frame_standard, stop_frame_standard, stop_frame_user_25,start_frame_user_25)
                if score_all>max_score_all:
                    max_score_all = score_all
                    max_score_limbs = score_limbs
                    # dict_match[stop_frame_standard] = stop_frame_user_25
                    match = [stop_frame_user_25, stop_frame_standard]

            list_matchs.append(match)
            score_limbs_all = max_score_limbs
            score_limbs_all[11] = max_score_all
            dict_stop_score[stop_frame_user_25] = score_limbs_all  # 匹配到了的用户停止帧就给他赋值
    return dict_stop_score, list_matchs

# def get_matched_score(matches, start_mixed_standard, fv_mul_front_standard, fpositions_left_user, fpositions_front_user,
#                           fpositions_left_standard, fpositions_front_standard):
#     """
#     获取所有匹配之后的停止帧的得分， 注意是以标准视频为依据，就是说：标准视频的每一个停止帧都会有对应的用户得分
#     :param matches:
#     :param start_mixed_standard: 双视角混合之后的标准视频的起始帧
#     :param fv_mul_front_standard: 标准视频前方摄像头的帧间向量积
#     :param fpositions_left_user: 用户的左边摄像头拍摄到针个视频的各个帧中所有关节的位置字典
#     :param fpositions_front_user: 用户的前方摄像头拍摄到针个视频的各个帧中所有关节的位置字典
#     :param fpositions_left_standard: 标准视频的左边摄像头拍摄到针个视频的各个帧中所有关节的位置字典
#     :param fpositions_front_standard: 标准视频的前方摄像头拍摄到针个视频的各个帧中所有关节的位置字典
#     :return: 一个dict{标准视频的某个动作的停止帧数：用户在这段动作内的得分（起始就是对应的停止帧的得分）}
#     :list_not_matches      : 一个list，标准视频中找不到对应帧的序号,比如在这里是[1170,1217]
#     """
#     # list_stop_standard = []
#     dict_stop_score = {}
#     list_not_matches=[]
#     for k in range(len(matches)):
#         if matches[k][0] == None:
#              all_score = 50
#              limbs_score = {i:0 for i in range(11)}
#              stop_frame_user=matches[k][1]
#              list_not_matches.append(stop_frame_user)
#         else:
#             start_frame = start_mixed_standard[k]
#             stop_frame_user, stop_frame_standard = matches[k]
#             limbs_score, all_score = get_all_scores(fpositions_front_standard, fpositions_left_standard, fpositions_front_user, fpositions_left_user, fv_mul_front_standard,
#                                                         start_frame, stop_frame_standard, stop_frame_user)
#         limbs_score[11] = all_score  # 将二者组合成一个dict
#         # list_stop_standard.append(stop_frame_standard)
#         dict_stop_score[stop_frame_user] = limbs_score
#     return dict_stop_score,list_not_matches
# 获取用户在这个标准视频的所有帧的得分

def get_video_score(total_frames_user, start_mixed_user, dict_score):
    """
    给整个视频的所有帧都提供一个分，在没有动作的地方给他空，也就是全零。 在一个动作的区间内（起始到结束）得分全为这个动作的结束帧匹配后比较得到的分数
    :param total_frames_standard: 标准视频流的总帧数
    :start_mixed_standard:标准的起始帧
    ：stop_mixed_standard:标准的结束帧
    : dict_score: 标准视频的各个停止帧的得分dict
    ：list_not_matches：一个list，标准视频中找不到对应帧的序号,比如在这里是[1170,1217]，对在用户视频该帧前后三帧输出50分
    :return: dict 每一个元素都是一个字典，对应着这个帧的各个关节的得分（0~10）以及总体得分（11）. 因此其长度和视频的总帧数一致
    """
    stop_s = [i for i in dict_score.keys()]
    start_all = start_mixed_user[0]  # 用户什么时候开始有动作
    scores = {}
    count = 0
    if len(stop_s)<1:
        for i in range(total_frames_user):
            score_wait = {i: 0 for i in range(11)}
            score_wait[11] = 0
            score = score_wait
            scores[i] = score
    else:
        for i in range(total_frames_user):
            # 用户还没开始第一个动作的时候就给他等待得分

            if i <= start_all:
                score_wait = {i: 0 for i in range(11)}
                score_wait[11] = 0
                score = score_wait
                scores[i] = score
            # 如果在一个动作的结束帧之前，那么就给他这个动作的得分
            elif ((i>start_all) and (i<=stop_s[count])):
                score = dict_score[stop_s[count]]
                scores[i] = score
            # 如果一个动作结束了而且后面还有动作，那么就切换到下一个动作的得分
            elif ((i>stop_s[count]) and (count<(len(stop_s)-1))):
                count += 1
                score = dict_score[stop_s[count]]
                scores[i] = score
            # 如果一个动作结束了，而且后面已经没有动作了，那么就给他等待得分
            else:
                score_wait = {i: 0 for i in range(11)}
                score_wait[11] = 0
                score = score_wait
                scores[i] = score
    return scores
# 根据躯干得分画躯干，画得分
def plot_eclipse(coord_start, coord_stop, image, color):
    center = ((coord_stop[0]+coord_start[0])//2, (coord_stop[1]+coord_start[1])//2)
    vec = [coord_stop[0]-coord_start[0], coord_stop[1]-coord_start[1]]
    length = int(np.linalg.norm(np.array(vec))//2)
    # print(length)
    width = 5
    vec_temp = [50, 0]
    cos_vec = (vec[0]*vec_temp[0] + vec[1]*vec_temp[1])/(np.linalg.norm(np.array(vec))*np.linalg.norm(np.array(vec_temp)))
    angle = 0
    if vec[1]<0:
        angle = math.acos(cos_vec)
    elif vec[1]>0:
        angle = math.acos(cos_vec)+math.pi
    else:
        if vec[0]>=0:
            angle = 0
        elif vec[0]<0:
            angle = 180
    # print(center)
    # print((length, width), 0, 0, -int((angle/math.pi)*180), color, 100)
    cv2.ellipse(image, center, (length, width), int((angle/math.pi)*180), 0, 360, color, -1)
    #
    # cv2.imshow("aa", image)
    # cv2.waitKey()
    return image
def draw_combine_image(image_user, image_standard, score, fpositions_user, fpositions_stantard, frame):
    """

    :param image_user: 待绘图的图像
    :param score: 这个图像的各个躯干以及最终得分
    :param fpositions_user:
    :param frame:
    :return:
    """
    xx = {0: [0, 1], 1: [2, 5], 2: [2, 3], 3: [5, 6], 4: [3, 4], 5: [6, 7], 6: [8, 11], 7: [8, 9], 8: [11, 12],
          9: [9, 10], 10: [12, 13], 11: [1, 8], 12: [1, 11]}
    dict_positions = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                      13: []}
    for key in range(14):  # 取出所有关节在frame这一帧的得分
        dict_positions[key] = fpositions_user[key][frame]
    for k in range(11):
        k1, k2 = xx[k]
        x1, y1 = dict_positions[k1]
        x2, y2 = dict_positions[k2]
        # print(frame)
        # print(score)
        score_limb = score[k]  # 获得某个躯干的得分
        if score_limb < 60:
            color = (205, 201, 122)
        elif 60 <= score_limb < 80:
            color = (74, 121, 255)
        elif 80 <= score_limb <= 100:
            color = (105, 107, 236)
        else:
            color = (0, 0, 255)
        plot_eclipse([int(x1), int(y1)], [int(x2), int(y2)], image_user, color)
        # cv2.line(image_user, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.putText(image_user, str(score[11]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 左上角协商得分
    for k in range(11, 13):
        k1, k2 = xx[k]
        x1, y1 = dict_positions[k1]
        x2, y2 = dict_positions[k2]
        cv2.line(image_user, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)


    # 给标准视频也画上躯干  只不过全是最低分
    for key in range(14):
        dict_positions[key] = fpositions_stantard[key][frame]
    for k in range(13):
        k1, k2 = xx[k]
        x1, y1 = dict_positions[k1]
        x2, y2 = dict_positions[k2]
        cv2.line(image_standard, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    # 左右拼接
    image_output = cv2.hconcat([image_standard, image_user])
    return image_output
def draw_combine_video(path_video_user, path_video_standard, path_save, fpositions_user, fpositions_stantard,
                       start_mixed_user, dict_score):
    """

    :param path_video_user: 将要绘制的用户视频
    :param path_video_stantard: 。。。与之拼接的标准视频
    :param fpositions_user: 用户各个帧的关节点位置
    :param fpositions_stantard: 标准视频各个帧的关节点位置
    :return: 直接保存
    """
    cap_user = cv2.VideoCapture(path_video_user)
    total_frames_user = int(cap_user.get(7))

    cap_standard = cv2.VideoCapture(path_video_standard)
    total_frames_standard = int(cap_standard.get(7))
    total_frames = min(total_frames_user, total_frames_standard)
    fps = cap_user.get(cv2.CAP_PROP_FPS)
    size = (int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH)*2), int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap_out = cv2.VideoWriter(path_save, fourcc, fps, size)
    scores = get_video_score(total_frames, start_mixed_user, dict_score)


    ret_val_user, image_user = cap_user.read()
    ret_val_standard, image_standard = cap_standard.read()
    count = 0
    while count<(total_frames-1):
        score = scores[count]
        image_output = draw_combine_image(image_user, image_standard, score, fpositions_user, fpositions_stantard, count)
        cap_out.write(image_output)
        ret_val_user, image_user = cap_user.read()
        ret_val_standard, image_standard = cap_standard.read()
        count += 1
    cap_standard.release()
    cap_user.release()
    cap_out.release()
    cv2.destroyAllWindows()
def main_score(path_video_user_front, path_video_user_left, path_json_user_front, path_json_user_left,
         path_video_standard_front, path_video_standard_left, path_json_standard_front, path_json_standard_left,
         path_video_output_front, path_video_output_left):
    fpositions_left_user = get_fposition(path_json_user_left)
    fpositions_front_user = get_fposition(path_json_user_front)
    fpositions_left_standard = get_fposition(path_json_standard_left)
    fpositions_front_standard = get_fposition(path_json_standard_front)
    start_frames_front_standard, stop_frames_front_standard, fv_mul_added_front_standard, fv_mul_front_standard = json2keyframes(path_json_standard_front)

    start_mixed_standard, stop_mixed_standard = mix_double_view(path_json_standard_left, path_json_standard_front)
    start_mixed_user, stop_mixed_user = mix_double_view(path_json_user_left, path_json_user_front)
    start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user = filter_mixed(start_mixed_standard,
                                                                                                stop_mixed_standard,
                                                                                                start_mixed_user,
                                                                                                stop_mixed_user)
    plot_lines(fv_mul_added_front_standard, start_mixed_standard, stop_mixed_standard)
    dict_score, matches = get_match_key_frame_25(start_mixed_standard, stop_mixed_standard,
                           start_mixed_user,stop_mixed_user,
                           fv_mul_front_standard,
                           fpositions_left_user, fpositions_front_user,
                           fpositions_left_standard, fpositions_front_standard)

    # 先画、拼接前方视角
    draw_combine_video(path_video_user_front, path_video_standard_front, path_video_output_front,
                       fpositions_front_user, fpositions_front_standard, start_mixed_user, dict_score)
    # 再画、拼接左侧视角
    draw_combine_video(path_video_user_left, path_video_standard_left, path_video_output_left,
                       fpositions_left_user, fpositions_left_standard, start_mixed_user, dict_score)


path_video_user_front = "K:\Pose_Estimate\\1A\\version1\data\\tizhuan_front_user5_63.avi"
path_video_user_left = "K:\Pose_Estimate\\1A\\version1\data\\tizhuan_left_user5_63.avi"
path_json_user_front = "K:\Pose_Estimate\\1A\\version1\\result\\tizhuan_front_user5_63.json"
path_json_user_left = "K:\Pose_Estimate\\1A\\version1\\result\\tizhuan_left_user5_63.json"

path_video_standard_front = "K:\Pose_Estimate\\1A\\version1\data\\tizhuan_front_standard5_63.avi"
path_video_standard_left = "K:\Pose_Estimate\\1A\\version1\data\\tizhuan_left_standard5_63.avi"
path_json_standard_front = "K:\Pose_Estimate\\1A\\version1\\result\\tizhuan_front_standard5_63.json"
path_json_standard_left = "K:\Pose_Estimate\\1A\\version1\\result\\tizhuan_left_standard5_63.json"

path_video_output_front = "K:\Pose_Estimate\\1A\\version1\data\output\\front.avi"
path_video_output_left = "K:\Pose_Estimate\\1A\\version1\data\output\\left.avi"
main_score(path_video_user_front, path_video_user_left, path_json_user_front, path_json_user_left,
         path_video_standard_front, path_video_standard_left, path_json_standard_front, path_json_standard_left,
         path_video_output_front, path_video_output_left)


