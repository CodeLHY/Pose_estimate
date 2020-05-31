from get_mean_score import *
from get_fvector_mul import *
import cv2
import os
# todo: shift还需要再考虑考虑
def search_match(stop_frames_user, stop_frames_standard, start_frames_standard):
    """
    以标准帧为基准，从用户的视频流中挑选出对应的帧，因此最后得到的match的总长度和标准视频中stop帧的个数是一样的
    :param stop_frames_user: 从用户的双视角综合出的所有停止帧
    :param stop_frames_standard: 从标准视频的双视角综合出的所有的停止帧
    :param start_frames_standard:  从表转换i品的双视角综合处的所有的起始帧
    :return: 匹配的结果构成的list 【用户帧， 标准帧】 如果用户帧为None，则表示标准流中这个停止帧没有在用户流中找到对应的结束帧
    """
    list_matches = []
    peace_areas_standard = get_peace_areas(start_frames_standard, stop_frames_standard)
    # shift = int(np.array(stop_frames_user).mean() - np.array(stop_frames_standard).mean())  # 用户的平均stop帧比标准视频中的平均stop帧快多少
    shift = 0
    index_user = 0
    index_standard = 0

    interval_pre_added = 0
    interval_nxt_added = 0 # 分别表示，停止帧相较于平静区的开始 和 结束 的距离的累加和
    interval_count = 0
    # 由于可能是刚好一个结束帧之后视频流便结束了，这时最后一个停止帧不存在对应的平静区，因此这里仅仅考虑前n-1个标准视频流的停止帧
    while ((index_standard<len(stop_frames_standard)-1) and index_user<len(stop_frames_user)):
        stop_frame_standard = stop_frames_standard[index_standard]
        stop_frame_user = stop_frames_user[index_user] - shift  # 减去整体的平移
        peace_area = peace_areas_standard[index_standard]
        relative = judge_in_area(stop_frame_user, peace_area)
        if relative == 0:
            # 说明当前的这个用户停止帧在 我们当前进行讨论的标准视频的停止帧对应的平静区的前面，这时我们直接不考虑这个用户的停止帧即可
            index_user += 1
            # 如果用户的结束帧都已经讨论完了 那么就直接break掉这个while循环
            if index_user>=(len(stop_frames_user)-1):
                break
        elif relative == 1:
            # 说明当前的这个用户停止帧在 我们当前进行讨论的标准视频的停止帧对应的平静区的内部，这时我们将他们作为一个匹配[用户帧， 标准帧]
            match = [stop_frame_user, stop_frame_standard]
            list_matches.append(match)

            index_user += 1
            index_standard += 1

            # 这些都是为了最后一个没有对应的平静区的停止帧做的打算
            interval_pre = stop_frame_standard - peace_area[0]
            interval_nxt = peace_area[1] - stop_frame_standard
            interval_pre_added += interval_pre
            interval_nxt_added += interval_nxt
            interval_count += 1
        else:
            # 说明当前的这个用户停止帧在 当前我们进行讨论的标准视频的停止帧对应的平静区的后面，这时我们认为用户没有跟上这个动作
            match = [None, stop_frame_standard]
            list_matches.append(match)
            index_standard += 1
    # while循环结束有两个原因： 1. 前n-1个标准视频的停止帧全部考虑完了，用户的停止帧也还没有讨论完
    #                     2. 用户的停止帧全部都考虑完了，但是标准的视频的还没有
    #                     3. 用户的停止帧全部都考虑完了 同时标准视频的前n-1个停止帧也都考虑完了
    if index_user == (len(stop_frames_user)-1):
        # 对应第2或第3种情况,这时，仅仅需要吧None和标准视频的停止帧进行匹配即可
        while(index_standard<len(stop_frames_standard)):
            stop_frame_standard = stop_frames_standard[index_standard]
            match = [None, stop_frame_standard]
            list_matches.append(match)
            index_standard += 1
    else:
        # 对应第1种情况，此时要先判断最后的一个停止帧是否有对应的平静区
        # 没有，那就计算平静区
        if len(stop_frames_standard)>len(peace_areas_standard):
            # 此时标准视频的最后一个停止帧也就是整个视频流的最后一个关键帧，因此没有对应的平静区
            mean_interval_pre = interval_pre_added//interval_count
            mean_interval_nxt = interval_nxt_added//interval_count
            stop_frame_standard = stop_frames_standard[index_standard]  # 获得标准视频的最后一个停止帧
            peace_area = [stop_frame_standard - mean_interval_pre, stop_frame_standard + mean_interval_nxt]  # 获得最后一个停止帧对应的平静区
        # 有，那就直接取出平静区
        else:
            stop_frame_standard = stop_frames_standard[index_standard]  # 获得标准视频的最后一个停止帧
            peace_area = peace_areas_standard[index_standard]
        flag_match_last = False  # 标记标准视频流中的最后一个停止帧是否已经获得了匹配
        while(index_user<len(stop_frames_user)):
            stop_frame_user = stop_frames_user[index_user] - shift  # 减去整体的平移
            relative = judge_in_area(stop_frame_user, peace_area)
            if relative == 0:
                # 说明当前的这个用户停止帧在 我们当前进行讨论的标准视频的停止帧对应的平静区的前面，这时我们直接不考虑这个用户的停止帧即可
                index_user += 1
                # 如果用户的结束帧都已经讨论完了 那么就直接break掉这个循环
                if index_user >= (len(stop_frames_user) - 1):
                    break
            elif relative == 1:
                # 说明当前的这个用户停止帧在 我们当前进行讨论的标准视频的停止帧对应的平静区的内部，这时我们将他们作为一个匹配[用户帧， 标准帧]
                match = [stop_frame_user, stop_frame_standard]
                list_matches.append(match)
                flag_match_last = True  # 如果找到了最后一个匹配，那么就break掉这个while
                break

            else:
                # 说明当前的这个用户停止帧在 当前我们进行讨论的标准视频的停止帧对应的平静区的后面，这时我们认为用户没有跟上这个动作
                match = [None, stop_frame_standard]
                list_matches.append(match)
                break
    return list_matches

def debug_plot_match(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user, matches):
    """

    :param start_mixed_standard:
    :param stop_mixed_standard:
    :param start_mixed_user:
    :param stop_mixed_user:
    :param matches:
    :return:  第一层是标准视频中起始帧和结束帧    第二层是用户视频的起始帧和结束帧     第三层是匹配出来的结束帧，黑色代表用户的，粉色代表标准的
    """
    plt.figure()
    for frame in start_mixed_standard:
        plt.vlines(frame, 0, 50, color="green")  # 竖线
    for frame in stop_mixed_standard:
        plt.vlines(frame, 0, 50, color="yellow")  # 竖线
    for frame in start_mixed_user:
        plt.vlines(frame, 50, 100, color="blue")  # 竖线
    for frame in stop_mixed_user:
        plt.vlines(frame, 50, 100, color="red")  # 竖线
    for match in matches:
        plt.pause(2)
        plt.vlines(match[0], 100, 150, color="black")  # 竖线
        plt.vlines(match[1], 100, 150, color="pink")  # 竖线
    plt.show()
def debug_save_match(matches, path_video_user, path_video_standard, save_dir):
    matched_user_frames = []
    matched_standard_frames = []
    for match in matches:
        if match[0] == None:
            continue
        else:
            matched_user_frames.append(match[0])
            matched_standard_frames.append(match[1])


    cap_user = cv2.VideoCapture(path_video_user)
    ret_val_user, image_user = cap_user.read()
    count_user = 0
    count_user_save = 0
    while ret_val_user:
        if count_user > 0:
            if count_user - 1 in matched_user_frames:

                name = save_dir+"/matched_"+str(count_user_save)+"_"+str(count_user)+"_user.jpg"
                if os.path.exists(name):
                    name = save_dir+"/matched_"+str(count_user_save)+"_"+str(count_user+1)+"_user.jpg"
                cv2.imwrite(name, image_user)
                count_user_save += 1
                if count_user == 355 or count_user == 565 or count_user == 831:
                    name = save_dir + "/matched_" + str(count_user_save) + "_" + str(count_user) + "_user.jpg"
                    cv2.imwrite(name, image_user)
                    count_user_save += 1
        ret_val_user, image_user = cap_user.read()
        count_user += 1
        cv2.waitKey(1)

    cap_standard = cv2.VideoCapture(path_video_standard)
    ret_val_standard, image_standard = cap_standard.read()
    count_standard = 0
    count_standard_save = 0
    while ret_val_standard:
        if count_standard > 0:
            if count_standard - 1 in matched_standard_frames:
                name = save_dir + "/matched_" + str(count_standard_save)+"_"+str(count_standard) + "_standard.jpg"
                cv2.imwrite(name, image_standard)
                count_standard_save += 1
        ret_val_standard, image_standard = cap_standard.read()
        count_standard += 1
        cv2.waitKey(1)
def debug_save_match2(matches, path_video_user, path_video_standard, save_dir):
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
        cv2.imwrite(name_user, list_image_user[match[0]])
        cv2.imwrite(name_standard, list_image_stantard[match[1]])


def filter_mixed(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user):
    """
    去除掉双视角混合之后可能出现的起始帧或者结束帧重复的情况，使得其成为101010的标准形式
    :param start_mixed_standard:
    :param stop_mixed_standard:
    :param start_mixed_user:
    :param stop_mixed_user:
    :return:
    """
    mixed_combined, mixed_property = list_combine(start_mixed_standard, stop_mixed_standard)
    mixed_combined, mixed_property = del_near_same_key(mixed_combined, mixed_property)
    start_mixed_standard, stop_mixed_standard = list_divide(mixed_combined, mixed_property)

    mixed_combined, mixed_property = list_combine(start_mixed_user, stop_mixed_user)
    mixed_combined, mixed_property = del_near_same_key(mixed_combined, mixed_property)
    start_mixed_user, stop_mixed_user = list_divide(mixed_combined, mixed_property)
    return start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user
"""
path_json_left_user = "./result/double/json526/xiadun/xiadun_left_user5_34.json"  #
path_video_left_user = "./data/double/data526/xiadun/xiadun_left_user5_34.avi"
path_json_front_user = "./result/double/json526/xiadun/xiadun_front_user5_34.json"  #
path_video_front_user = "./data/double/data526/xiadun/xiadun_front_user5_34.avi"
start_frames_left_user, stop_frames_left_user, fv_mul_added_left_user = json2keyframes(path_json_left_user)
start_frames_front_user, stop_frames_front_user, fv_mul_added_front_user = json2keyframes(path_json_front_user)

path_json_left_standard = "./result/double/json526/xiadun/xiadun_left_standard5_34.json"  #
path_video_left_standard = "./data/double/data526/xiadun/xiadun_left_standard5_34.avi"
path_json_front_standard = "./result/double/json526/xiadun/xiadun_front_standard5_34.json"  #
path_video_front_standard = "./data/double/data526/xiadun/xiadun_front_standard5_34.avi"
start_frames_left_standard, stop_frames_left_standard, fv_mul_added_left_standard = json2keyframes(path_json_left_standard)
start_frames_front_standard, stop_frames_front_standard, fv_mul_added_front_standard = json2keyframes(path_json_front_standard)



# path_json_left_user = "./result/double/xiadun/xiadun_left_user4_32.json"  #
# path_video_left_user = "./data/double/xiadun/xiadun_left_user4_32.avi"
# path_json_front_user = "./result/double/xiadun/xiadun_front_user4_32.json"  #
# path_video_front_user = "./data/double/xiadun/xiadun_front_user4_32.avi"
# start_frames_left_user, stop_frames_left_user, fv_mul_added_left_user = json2keyframes(path_json_left_user)
# start_frames_front_user, stop_frames_front_user, fv_mul_added_front_user = json2keyframes(path_json_front_user)
#
# path_json_left_standard = "./result/double/xiadun/xiadun_left_standard4_32.json"  #
# path_video_left_standard = "./data/double/xiadun/xiadun_left_standard4_32.avi"
# path_json_front_standard = "./result/double/xiadun/xiadun_front_standard4_32.json"  #
# path_video_front_standard = "./data/double/xiadun/xiadun_front_standard4_32.avi"
# start_frames_left_standard, stop_frames_left_standard, fv_mul_added_left_standard = json2keyframes(path_json_left_standard)
# start_frames_front_standard, stop_frames_front_standard, fv_mul_added_front_standard = json2keyframes(path_json_front_standard)

start_mixed_standard, stop_mixed_standard = mix_double_view(path_json_left_standard, path_json_front_standard)
start_mixed_user, stop_mixed_user = mix_double_view(path_json_left_user, path_json_front_user)
start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user = filter_mixed(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user)
matches = search_match(stop_mixed_user, stop_mixed_standard, start_mixed_standard)




debug_save_match(matches, path_video_front_user, path_video_front_standard, "K:\Pose_Estimate\\1A\Key_frame\key_images\matches")
plot_double_ss_compare(start_mixed_user, stop_mixed_user,start_mixed_user, stop_mixed_user,
                           start_frames_front_user, stop_frames_front_user,start_frames_left_user, stop_frames_left_user)


key_images_standard, key_property_standard = get_key_image(path_video_front_standard, start_mixed_standard, stop_mixed_standard)
key_images_user, key_property_user = get_key_image(path_video_front_user, start_mixed_user, stop_mixed_user)
dir_save_standard = "./key_images/standard"
dir_save_user = "./key_images/user"
save_key_images(key_images_standard, key_property_standard, dir_save_standard)
save_key_images(key_images_user, key_property_user, dir_save_user)


debug_plot_match(start_mixed_standard, stop_mixed_standard, start_mixed_user, stop_mixed_user, matches)
#  标准视频使用起始帧和结束帧从而可以计算这个动作的权重，实际比较的时候仅仅比较结束帧
#  用户的结束帧选取规则：标准视频的结束帧先平移α帧， 然后在对应的前后X帧的范围内寻找，超过这个范围就是找不到这个结束帧
#  α的选取规则为：起始和结束之间的长度的100分之1，具体多少可能要根据实验来得出
#  X的选取规则为：本次结束帧和下一个起始帧之间的距离的范围内寻找，因为当时是让  这个结束帧和下一个起始帧分别后移和前移了1/3他们之间的距离，
#  因此如果用户真的跟着做了这个动作的话针个范围应该足够了。当然还要定义一个结束帧之间帧数之差的函数，作为最终得分的权重。为了不打击用户的自信心
#  我们应该定义：只要检测到了结束帧 最低分也不能为0，
"""