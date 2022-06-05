
import numpy as np
import pandas as pd
import warnings
import time
import time
import random
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# from lightgbm import LGBMClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import csv

warnings.filterwarnings("ignore")  # ignore warning
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

'''
Divide equivalence classes
atts：attribute set
data：data set
return：Returns the equivalence class, a two-dimensional list
'''
def eql_class_split(not_redu, current_equ_classes, data):
    ud_equ_classes = []
    for equ_class in current_equ_classes:  # Add the most important properties, update the equivalence class
        new_equ_classes = {}
        for sample in equ_class:
            if data[sample, not_redu] not in new_equ_classes:
                new_equ_classes[data[sample, not_redu]] = [sample]
            else:
                new_equ_classes[data[sample, not_redu]].append(sample)
        for keys in new_equ_classes:
            ud_equ_classes.append(new_equ_classes[keys])
    return ud_equ_classes

'''
attribute reduction
'''
def att_rdtion(data):
    current_equ_class = [[i for i in range(len(data))]]  # current_equ_class is used to save the current equivalence class, and the entire dataset is initially an equivalence class
    not_redund_atts = []    #Save Reduced Properties
    if len(data) != 0:
        redund_atts = [i for i in range(len(data[0, :-2]))]  # index list of all attributes
        active_re = {}  # holds the equivalence class for each attribute

        for redund_att in redund_atts:
            result1 = eql_class_split(redund_att, current_equ_class, data)
            active_re[redund_att] = result1
        active_re_keys = active_re.keys()
        active_re_keys = list(active_re_keys)

        active_list111 = {}
        # Get the smallest sample number in a single attribute equivalence class
        for index in active_re_keys:
            list444 = []
            for item in active_re[index]:
                list444.append(item[0])
            active_list111[index] = list444
        #Calculate the first property. Because the non-active region is empty when the first attribute is calculated, it is calculated directly on the entire data set at this time.
        max_num1 = 0
        prepare_att1 = -1
        for index_1 in active_re_keys:
            re_list = []
            num_re_111 = 0
            for new_item in active_re[index_1]:
                list11 = []  # used to hold decision attributes for each equivalence class
                for att_num in new_item:
                    list11.append(data[att_num][-2])
                if len(set(list11)) == 1:
                    re_list.append(new_item)
                    num_re_111 += len(new_item)
            if num_re_111 > max_num1:
                max_num1 = num_re_111
                prepare_att1 = index_1
        redund_atts.remove(prepare_att1)
        active_re_keys.remove(prepare_att1)
        not_redund_atts.append(prepare_att1)
        current_equ_class = active_re[prepare_att1]
        #attribute reduction
        while redund_atts:
            stemp_eql_class = {}  # Used to save the temporary equivalence class generated when finding the positive region
            list_item = []
            for equ_class in current_equ_class:  # find boundary region
                list_split = []
                for index in equ_class:
                    list_split.append(data[index][-2])
                if len(set(list_split)) == 1:
                    list_item.append(equ_class)

            if len(list_item) > 0:
                for item in list_item:
                    current_equ_class.remove(item)
            board_area = current_equ_class.copy()  # get bounding region
            prepare_att = -1
            max_num = 0

            for index in active_re_keys:
                lists = active_re[index]  # Equivalence class under each attribute
                board_area_stemp = board_area.copy()  # Temporary storage of equivalence classes, saving the current active region
                active_list = []  # Save equivalence classes for non-active region

                if (2 * len(board_area_stemp) < len(active_re[index])):
                    num_re_111 = 0  # The number of new positive region under the current attribute
                    new_eql_class = eql_class_split(index, board_area_stemp, data)  # new_eql_class is used to save the new equivalence class after adding an attribute
                    stemp_eql_class[index] = new_eql_class.copy()
                    num_class = len(new_eql_class) - len(board_area_stemp)  # The number of newly added equivalence classes
                    for new_item in new_eql_class:
                        list11 = []  #used to hold decision attributes for each equivalence class
                        for att_num in new_item:
                            list11.append(data[att_num][-2])
                        if len(set(list11)) == 1:
                            num_re_111 += len(new_item)
                    if num_class != 0:       #Calculate the relative importance of attributes
                        if num_re_111 / num_class> max_num:
                            max_num = num_re_111/ num_class
                            prepare_att = index
                else:
                    for item in board_area_stemp:  # This layer of loop is used to find the active region
                        num2 = serarch(active_list111[index], item[0])
                        k = 0
                        l = binarySearch(lists[num2], 0, len(lists[num2]) - 1, item[0])
                        r = binarySearch(lists[num2], 0, len(lists[num2]) - 1, item[len(item) - 1])

                        if len(item) > r - l + 1:
                            break
                        if l <= -1 or r <= -1:
                            break
                        else:
                            for i in range(1, len(item) - 1):
                                m = binarySearch(lists[num2], l, r, item[i])
                                if m > -1:
                                    k += 1
                                else:
                                    break
                        if k == len(item) - 2:  # The first and last elements have been judged
                            active_list.append(item)
                            del lists[num2][l:r+1]
                    if len(active_list) > 0:
                        for item in active_list:
                            board_area_stemp.remove(item)
                    if len(board_area_stemp) == 0:
                        continue
                    num_re_111 = 0  # The number of new positive region under the current attribute
                    new_eql_class = eql_class_split(index, board_area_stemp, data)  # new_eql_class is used to save the new equivalence class after adding an attribute
                    stemp_eql_class[index] = new_eql_class.copy()

                    num_class = len(new_eql_class) - len(board_area_stemp)  # The number of newly added equivalence classes
                    for new_item in new_eql_class:
                        list11 = []  # used to hold decision attributes for each equivalence class
                        for att_num in new_item:
                            list11.append(data[att_num][-2])
                        if len(set(list11)) == 1:
                            num_re_111 += len(new_item)
                    #Calculate the relative importance of attributes
                    if num_class!=0:
                        if num_re_111/ num_class > max_num:
                            max_num = num_re_111/ num_class
                            prepare_att = index
                            for iii in active_list:
                                stemp_eql_class[index].append(iii)
            if prepare_att < 0:  # prepare_att<0 means that all executions do not increase the positive region and stop the loop
                break
            redund_atts.remove(prepare_att)
            current_equ_class = stemp_eql_class[prepare_att].copy()
            active_re_keys.remove(prepare_att)
            not_redund_atts.append(prepare_att)
    print('not_redund_atts=======>', sorted(not_redund_atts))
    return not_redund_atts


def binarySearch(arr, l, r, x):
    left = 0
    right = r
    while left <= right:
        mid = (left + right) // 2
        if x < arr[mid]:
            right = mid - 1
        elif x > arr[mid]:
            left = mid + 1
        else:
            return mid
    return -2

'''
binary search
lis：target set
num：target number
'''
def serarch(lis, num):
    left = 0
    right = len(lis) - 1
    while left <= right:
        # if (right - 1 == left):
        #     return left
        mid = left+(right-left) // 2
        if num < lis[mid]:
            right = mid - 1
        elif num > lis[mid]:
            left = mid + 1
        else:
            return mid
    return left - 1

'''
Calculate the mean and standard deviation of a 1D array
'''
def mean_std(a):
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std

'''
Rough concept tree for classification
re：reduction result
data：data set
train_data：Training set
'''
def classifier(re, data, train_data):
    eql_class = [[i for i in range(len(data))]]
    for index in re:
        eql_class = eql_class_split(index, eql_class, data)
    atts_dict = judge_re(re, eql_class, data)
    atts_dict_keylist11 = sorted(list(atts_dict.keys()))

    right_num = 0  # Indicates the correct number of samples to predict
    for train_sample in train_data:
        list44 = []
        for index in re:
            list44.append(train_sample[index])
        atts_dict_keylist11.append(str(list44))
        atts_dict_keylist = sorted(atts_dict_keylist11).copy()

        train_location = sorted(atts_dict_keylist).index(str(list44))
        if train_location + 1 > len(atts_dict_keylist) - 1:
            train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
        else:
            if atts_dict_keylist[train_location - 1] == atts_dict_keylist[train_location]:
                train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
            elif atts_dict_keylist[train_location + 1] == atts_dict_keylist[train_location]:
                train_decision = atts_dict[atts_dict_keylist[train_location + 1]]
            else:
                for i in range(min(len(atts_dict_keylist[train_location]), len(atts_dict_keylist[train_location - 1]),
                                   len(atts_dict_keylist[train_location + 1]))):
                    if atts_dict_keylist[train_location - 1][i] == atts_dict_keylist[train_location][i] and \
                            atts_dict_keylist[train_location][i] == atts_dict_keylist[train_location + 1][i]:
                        continue
                    else:
                        if ord(atts_dict_keylist[train_location][i]) - ord(
                                atts_dict_keylist[train_location - 1][i]) < ord(
                                atts_dict_keylist[train_location + 1][i]) - ord(atts_dict_keylist[train_location][i]):
                            train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
                        else:
                            train_decision = atts_dict[atts_dict_keylist[train_location + 1]]
        if train_decision == train_sample[-2]:
            right_num += 1
        atts_dict_keylist11.remove(str(list44))
    acc = right_num / len(train_data)
    print(acc)
    return acc

def judge_re(re,eql_class,data):
    atts_dict ={}      #Used to save the decision attributes corresponding to each attribute set
    for new_item in eql_class:
        list22 =[]
        for index in re:
            list22.append(data[new_item[0]][index])

        list11 = []  #used to hold decision attributes for each equivalence class
        for att_num in new_item:
            list11.append(data[att_num][-2])
        atts_dict[str(list22)] =max(list11, key=list11.count)

    return atts_dict

classifiers = ['LR'] #Classifier used
keys = ["Amazon_initial_50_30_10000"]  #data set used
for key in keys:
    df = pd.read_csv("D:\\path", header=None)
    data = df.values
    numberSample, numberAttribute =data.shape

    #Data discretization
    # for i in range(1, len(data[0])):
    #     start1_time = time.clock()
    #     # print(i)
    #     if len(set(data[:, i])) == 1:
    #         continue
    #     group = KF.Chi_Discretization(df, i, 0, max_interval=10, binning_method='chiMerge', feature_type=0)  # binning
    #     for j in range(len(data)):  # Data replacement Direct replacement to the original data
    #         k = 0
    #         while (True):
    #             # print(k)
    #             if (k == len(group) - 1):
    #                 data[j][i] = group[k]
    #                 # new_data[j][0]=data.values[j][0]
    #                 break
    #             if (data[j][i] < group[k + 1] and data[j][i] >= group[k]):
    #                 data[j][i] = group[k + 1]
    #                 # new_data[j][0] = data.values[j][0]
    #                 break
    #             else:
    #                 k += 1


    data = np.hstack((data[:, 1:], data[:, 0].reshape(numberSample, 1)))

    orderAttribute = np.array([i for i in range(0, numberSample)]).reshape(numberSample,
                                                                           1)  # Create a list holding the sequence of numbers from 1 to numberSample
    data = np.hstack((data, orderAttribute))
    a = time.clock()
    re = att_rdtion(data)
    b = time.clock()
    print("time===========>", b - a)


    for c in range(len(classifiers)):
        classifier = classifiers[c]
        print(classifier)
        if classifier == 'BP':
            clf = MLPClassifier()
        elif classifier == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier == 'SVM':
            clf = SVC(kernel='rbf', gamma='auto')
        elif classifier == 'CART':
            clf = DecisionTreeClassifier()
        elif classifier == 'LR':
            clf = LogisticRegression(solver='liblinear')
        elif classifier == 'GBDT':
            clf = GradientBoostingClassifier()

        if re != [i for i in range(len(data[0, :-1]))]:
            random.shuffle(data)
            mat_data = data[:, re]
            start_time2 = time.clock()
            orderAttribute = data[:, -2]
            scores = cross_val_score(clf, mat_data, orderAttribute, cv=10)
            avg1, std1 = mean_std(scores)
            print("acc", avg1, "\t", std1)
            end_time2 = time.clock()
            run_time2 = end_time2 - start_time2
            result = []
            result.append(key)
            result.append(format(avg1, '.4f') + '±' + format(std1, '.4f'))
            with open(r'E:\\path', 'a+', encoding='utf-8-sig',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
        else:
            print("This time is not reduced")
            break









