import pandas as pd
import numpy as np
import json
import os
import sys
import mmcv


diff_flag = sys.argv[1:]


# iou的分母为物品，即交叉框占物品面积的比重
def obj_iou(preson_frame, thing_frame):
    x_min = max(preson_frame[0], thing_frame[0])
    y_min = max(preson_frame[1], thing_frame[1])
    x_max = min(preson_frame[2], thing_frame[2])
    y_max = min(preson_frame[3], thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    iou = intersection / thing
    return iou


# iou的分母为人，即交叉框占人面积的比重
def obj_iou_on_person(preson_frame, thing_frame):
    x_min = max(preson_frame[0], thing_frame[0])
    y_min = max(preson_frame[1], thing_frame[1])
    x_max = min(preson_frame[2], thing_frame[2])
    y_max = min(preson_frame[3], thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    person = (preson_frame[2] - preson_frame[0]) * (preson_frame[3] - preson_frame[1])
    iou = intersection / person
    return iou


# 判断两个人是否有一个人完全覆盖另一个人
def iou_2_person(person1, person2):
    x_min = max(person1[0], person2[0])
    y_min = max(person1[1], person2[1])
    x_max = min(person1[2], person2[2])
    y_max = min(person1[3], person2[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return False, 3

    intersection = (x_max - x_min) * (y_max - y_min)
    person1_scale = (person1[2] - person1[0]) * (person1[3] - person1[1])
    person2_scale = (person2[2] - person2[0]) * (person2[3] - person2[1])
    iou1 = intersection / person1_scale
    iou2 = intersection / person2_scale
    if iou1 > 0.99999:
        return True, 2
    if iou2 > 0.99999:
        return True, 1

    return False, 3


# 判断两个安全带是否有一个完全覆盖另一个，阈值0.5
def iou_2_belt(belt1, belt2):
    x_min = max(belt1[0], belt2[0])
    y_min = max(belt1[1], belt2[1])
    x_max = min(belt1[2], belt2[2])
    y_max = min(belt1[3], belt2[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return False, 3

    intersection = (x_max - x_min) * (y_max - y_min)
    belt1_scale = (belt1[2] - belt1[0]) * (belt1[3] - belt1[1])
    belt2_scale = (belt2[2] - belt2[0]) * (belt2[3] - belt2[1])
    iou1 = intersection / belt1_scale
    iou2 = intersection / belt2_scale
    if iou1 > 0.5:
        return True, 2
    if iou2 > 0.5:
        return True, 1

    return False, 3
    # if iou1 > 0.99999 or iou2 > 0.99999:
    #     return True
    # else:
    #     return False


# 求勋章的iou
def badge_iou(preson_frame, thing_frame, p=0.9):
    x_min = max(preson_frame[0], thing_frame[0])
    y_min = max(preson_frame[1], thing_frame[1])
    x_max = min(preson_frame[2], thing_frame[2])
    y_max = min(preson_frame[3], thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return False

    intersection = (x_max - x_min) * (y_max - y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    iou = intersection / thing
    # print(iou)
    if iou >= p:
        return True
    else:
        return False


# 求安全带的iou
def safebelt_iou(preson_frame, thing_frame, p=0.3):
    x_min = max(preson_frame[0], thing_frame[0])
    y_min = max(preson_frame[1], thing_frame[1])
    x_max = min(preson_frame[2], thing_frame[2])
    y_max = min(preson_frame[3], thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return False

    intersection = (x_max - x_min) * (y_max - y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    iou = intersection / thing
    # print(iou)
    if iou >= p:
        return True
    else:
        return False


if __name__ == '__main__':
    test_csv_name = 'D:/137/dataset/gdgrid3/3_testb_imageid.csv'
    result_dir = 'work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco'
    result_pkl_name = os.path.join(result_dir, 'results.pkl')
    result_json_name = os.path.join(result_dir, 'results.json')
    
    df = pd.read_csv(test_csv_name, header=0)
    df = df["image_url"]
    
    
    pkl_results = mmcv.load(result_pkl_name)
    
    results = []  # 提交结果

    for id_s, one_img_name in enumerate(df):
        one_img_name = one_img_name.split("/")[-1].split(".")[0] + ".txt"
        one_img_result = pkl_results[id_s]
        if isinstance(one_img_result, tuple):
            one_img_result = one_img_result[0]
        
        offground, ground = [], []  # 人
        badge, safebelt = [], []  # 物体
        badge_people = []
        belt_people = []
        
        for cat, cat_res in enumerate(one_img_result):
            if cat == 1:
                for one_res in cat_res:
                    offground.append(np.array([int(one_res[0]), int(one_res[1]),
                                               int(one_res[2]), int(one_res[3]), one_res[4]]))
            elif cat == 2:
                for one_res in cat_res:
                    ground.append(np.array([int(one_res[0]), int(one_res[1]),
                                               int(one_res[2]), int(one_res[3]), one_res[4]]))
            elif cat == 0:
                for one_res in cat_res:
                    badge.append(np.array([int(one_res[0]), int(one_res[1]),
                                               int(one_res[2]), int(one_res[3]), one_res[4]]))
            elif cat == 3:
                for one_res in cat_res:
                    safebelt.append(np.array([int(one_res[0]), int(one_res[1]),
                                               int(one_res[2]), int(one_res[3]), one_res[4]]))
            else:
                raise ValueError('cat should be in [0, 1, 2, 3]')
        
        # one_txt = one_txt[0]
        
        # for one_res in one_txt:
        #     one_res = one_res.split(" ")
        #     if int(one_res[0]) == 2:
        #         frame = np.array([one_res[1], one_res[2], one_res[3], one_res[4], one_res[5]]).astype(np.float32)
        #         offground.append(np.array(
        #             [int(frame[1]), int(frame[2]), int(frame[1]) + int(frame[3]), int(frame[2]) + int(frame[4]), frame[0]]))
        #     elif int(one_res[0]) == 3:
        #         frame = np.array([one_res[1], one_res[2], one_res[3], one_res[4], one_res[5]]).astype(np.float32)
        #         ground.append(np.array(
        #             [int(frame[1]), int(frame[2]), int(frame[1]) + int(frame[3]), int(frame[2]) + int(frame[4]), frame[0]]))
        #     elif int(one_res[0]) == 1:
        #         frame = np.array([one_res[1], one_res[2], one_res[3], one_res[4], one_res[5]]).astype(np.float32)
        #         badge.append(np.array(
        #             [int(frame[1]), int(frame[2]), int(frame[1]) + int(frame[3]), int(frame[2]) + int(frame[4]), frame[0]]))
        #     elif int(one_res[0]) == 4:
        #         frame = np.array([one_res[1], one_res[2], one_res[3], one_res[4], one_res[5]]).astype(np.float32)
        #         safebelt.append(np.array(
        #             [int(frame[1]), int(frame[2]), int(frame[1]) + int(frame[3]), int(frame[2]) + int(frame[4]), frame[0]]))
        #     else:
        #         break
        
        # 全部的人
        all_people = offground + ground

        # 判断两个belt如果完全重合，则删除小面积的那个
        if "1" not in diff_flag:
            delete_list = []
            for index1 in range(len(safebelt)):
                if index1 in delete_list:
                    continue
                for index2 in range(index1 + 1, len(safebelt)):
                    if index2 in delete_list:
                        continue
                    flag, who = iou_2_belt(safebelt[index1][0:4], safebelt[index2][0:4])
                    if who == 1:
                        if index1 not in delete_list:
                            delete_list.append(index1)
                    elif who == 2:
                        if index2 not in delete_list:
                            delete_list.append(index2)

            if len(safebelt) > 1:
                delete_list.sort()
                for index,i in enumerate(delete_list):
                    del safebelt[i-index]

        # 遍历每个belt，然后递归每个人，找出belt交叉区域占belt面积最大的那个人为belt人
        for belt in safebelt:
            belt_frame = belt[0:4]
            max_index = -1
            max_iou = 0.0

            for index, off in enumerate(all_people):
                belt_iou = obj_iou(off[0:4], belt_frame)
                if obj_iou(off[0:4], belt_frame) > max_iou:
                    max_iou = obj_iou(off[0:4], belt_frame)
                    max_index = index
            if max_index != -1:
                belt_people.append(all_people[max_index])
                # result = {}
                # result["image_id"] = id_s
                # result["category_id"] = 2
                # result["bbox"] = [all_people[max_index][0], all_people[max_index][1], all_people[max_index][2],
                #                   all_people[max_index][3]]
                # result["score"] = float(all_people[max_index][4])
                # results.append(result)


        # 遍历每个arm，然后递归每个人，找出arm交叉区域占arm面积最大的那个人为arm人
        for arm in badge:
            arm_frame = arm[0:4]
            max_index = -1
            max_iou = 0.0
            for index, off in enumerate(all_people):
                arm_iou = obj_iou(off[0:4], arm_frame)
                if "2" not in diff_flag:
                    if arm_iou > 0.99999 and max_iou > 0.99999:
                        height_origin = abs(all_people[max_index][1] - arm_frame[1])
                        height_now = abs(off[1] - arm_frame[1])
                        if height_now < height_origin:
                            max_iou = arm_iou
                            max_index = index
                            continue
                if arm_iou > max_iou:
                    max_iou = arm_iou
                    max_index = index
            if max_index != -1:
                badge_people.append(all_people[max_index])
                # result = {}
                # result["image_id"] = id_s
                # result["category_id"] = 1
                # result["bbox"] = [all_people[max_index][0], all_people[max_index][1], all_people[max_index][2],
                #                   all_people[max_index][3]]
                # result["score"] = float(all_people[max_index][4])
                # results.append(result)

        # 判断两个离地的人如果完全重合，则删除小面积的那个
        if "3" not in diff_flag:
            delete_list = []
            for index1 in range(len(offground)):
                if index1 in delete_list:
                    continue
                for index2 in range(index1 + 1, len(offground)):
                    if index2 in delete_list:
                        continue
                    flag, who = iou_2_person(offground[index1][0:4], offground[index2][0:4])
                    if flag == True:
                        if who == 2:
                            if index1 not in delete_list:
                                delete_list.append(index1)
                        elif who == 1:
                            if index2 not in delete_list:
                                delete_list.append(index2)

            if len(offground) > 1:
                delete_list.sort()
                for index,i in enumerate(delete_list):
                    del offground[i-index]

        # 判断两个安全带人如果完全重合，则删除小面积的那个
        if "4" not in diff_flag:
            delete_list = []
            for index1 in range(len(belt_people)):
                if index1 in delete_list:
                    continue
                for index2 in range(index1 + 1, len(belt_people)):
                    if index2 in delete_list:
                        continue
                    flag, who = iou_2_person(belt_people[index1][0:4], belt_people[index2][0:4])
                    if flag == True:
                        if who == 2:
                            if index1 not in delete_list:
                                delete_list.append(index1)
                        elif who == 1:
                            if index2 not in delete_list:
                                delete_list.append(index2)

            if len(belt_people) > 1:
                delete_list.sort()
                for index,i in enumerate(delete_list):
                    del belt_people[i-index]

        # 判断两个袖章人如果完全重合，则删除小面积的那个
        if "5" not in diff_flag:
            delete_list = []
            for index1 in range(len(badge_people)):
                if index1 in delete_list:
                    continue
                for index2 in range(index1 + 1, len(badge_people)):
                    if index2 in delete_list:
                        continue
                    flag, who = iou_2_person(badge_people[index1][0:4], badge_people[index2][0:4])
                    if flag == True:
                        if who == 2:
                            if index1 not in delete_list:
                                delete_list.append(index1)
                        elif who == 1:
                            if index2 not in delete_list:
                                delete_list.append(index2)

            if len(badge_people) > 1:
                delete_list.sort()
                for index,i in enumerate(delete_list):
                    del badge_people[i-index]

        # 其余离地的人加入结果
        for off in offground:
            result = {}
            result["image_id"] = id_s
            result["category_id"] = 3
            result["bbox"] = [off[0], off[1], off[2], off[3]]
            result["score"] = float(off[4])
            results.append(result)

        # 其余安全带的人加入结果
        for off in belt_people:
            result = {}
            result["image_id"] = id_s
            result["category_id"] = 2
            result["bbox"] = [off[0], off[1], off[2], off[3]]
            result["score"] = float(off[4])
            results.append(result)
        # 其余袖章的人加入结果
        for off in badge_people:
            result = {}
            result["image_id"] = id_s
            result["category_id"] = 1
            result["bbox"] = [off[0], off[1], off[2], off[3]]
            result["score"] = float(off[4])
            results.append(result)
    
    print(len(results))
    json.dump(results, open(result_json_name, 'w'), indent=4)
