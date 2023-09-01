#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob, os
import json
import copy
import math
from pandas.io.parsers import read_csv
from lxml import etree


def get_classname_from_xpath(xpath):
    return xpath.split('//')[1].split('[')[0]


def get_attribute_from_xpath(xpath):
    if '""' in xpath:
        xpath = xpath.replace('""', '"')
    return xpath[xpath.find("[") + 1:xpath.find("]")]

def find_node_by_xpath(xpath, app):
    directory = 'input1/screenshots/shopping/' + app + '/'
    # print('find node for xpath', xpath, 'in app', app)
    for filename in os.listdir(directory):
        if filename.endswith(".uix"):
            # print('check xpath in ', os.path.join(directory, filename))
            tree = etree.parse(os.path.join(directory, filename))
            root = tree.getroot()
            if xpath.startswith('//'):  # relative xpath
                class_name = get_classname_from_xpath(xpath)
                attribute = get_attribute_from_xpath(xpath)
                # print('//node[@class="'+class_name+'"]['+attribute+']')
                nodes = root.xpath('//node[@class="' + class_name + '"][' + attribute + ']')
                if len(nodes) != 0:
                    # print('current node is ', etree.tostring(nodes[0]))
                    return nodes[0]
            elif xpath.startswith('/hierarchy'):  # absolute xpath
                class_names = xpath.split('/')
                # print(class_names)
                current_node = root.xpath('/hierarchy')[0]
                no_matching = False
                for class_name in class_names:
                    if class_name == '' or class_name == 'hierarchy':
                        continue
                    # print('.//node[@class="' + class_name +'"]')
                    if '[' in class_name:  # multiple children with same class name
                        index = int(class_name[class_name.find("[") + 1:class_name.find("]")])
                        class_name = class_name.split('[')[0]
                        current_nodes = current_node.findall('./node[@class="' + class_name + '"]')
                        if current_nodes is None or index >= len(current_nodes):
                            no_matching = True
                            break
                        else:
                            current_node = current_nodes[index]
                    else:  # only one child with same class name
                        current_nodes = current_node.findall('./node[@class="' + class_name + '"]')
                        if current_nodes is None or len(current_nodes) == 0:
                            no_matching = True
                            break
                        else:
                            current_node = current_nodes[0]
                if not no_matching:
                    # print('current node is ', etree.tostring(current_node))
                    return current_node
    # print('current node is None')
    return None

# trans test format: json with "input", "id_or_xpath", "action", "case". 'id_or_xpath' could be 'NONE'
# gt test format: 'id@...'/'xpath@...'
def trans_equals_gt(trans_event, gt_event, tgt_app):
    # when trans and gt use the same id or xpath
    trans_id_or_xpath = trans_event['id_or_xpath'] 
    if gt_event == trans_id_or_xpath:
        return True
    if gt_event[:3] == "id@": # gt_event is based on resource-id
        if trans_id_or_xpath[:3] == "id@":
            return False
        else: # gt uses id and trans uses xpath
            return compare_id_xpath(gt_event[3:], trans_id_or_xpath[6:], tgt_app)
    else: # gt_event is based on xpath
        if trans_id_or_xpath[:3] == "id@": # trans uses id, gt uses xpath
            return compare_id_xpath(trans_id_or_xpath[3:], gt_event[6:], tgt_app)
        else: # both gt and trans use xpath. one could use absolute xpath and another one uses relevant xpath
            gt_node = find_node_by_xpath(gt_event[6:], tgt_app)
            trans_node = find_node_by_xpath(trans_id_or_xpath[6:], tgt_app)
            return (gt_node == trans_node)

def compare_id_xpath(id, xpath, app):
    node = find_node_by_xpath(xpath, app) 
    if node is not None and id == node.get('resource-id'):
        return True
    return False

# trans test format: json with "input", "id_or_xpath", "action", "case". 'id_or_xpath' could be 'NONE'
# gt test format: 'id@...'/'xpath@...'
# return the levenshtein distance
def levenshtein(test):
    transfer_seq = test['event_array']
    gt_seq = test['gt_events']
    result = {}
    # if gt test doesn't exist in the target app, return NA
    if type(gt_seq) is float and math.isnan(gt_seq):
        # print(gt_seq)
        result['distance'] = np.NaN
        return result
    
    trans = copy.deepcopy(transfer_seq)
    trans = json.loads(trans)
    gt = copy.deepcopy(gt_seq)
    
    # delete 'NONE' events in order to calculate levenshtein distance correctly
    # print('before trans = ', trans)
    none_events = []
    for event in trans:
        if event['id_or_xpath'] == 'NONE':
            none_events.append(event)
    for event in none_events:
        trans.remove(event)
    # print('trans = ', trans)
    # print('gt = ', gt)

    size_x = len(trans) + 1
    size_y = len(gt) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if trans_equals_gt(trans[x-1], gt[y-1], test['target']):
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    # print('distance = ', (matrix[size_x - 1, size_y - 1]))
    result['distance'] =  (matrix[size_x - 1, size_y - 1])
    return result

app_name_mapping = {}
app_name_mapping['lightning'] = 'Lightning'
app_name_mapping['browser'] = 'Browser'
app_name_mapping['privacy'] = 'Privacy'
app_name_mapping['foss'] = 'Foss'
app_name_mapping['firefox'] = 'Firefox'

app_name_mapping['minimal'] = 'Minimal'
app_name_mapping['clear'] = 'Clear'
app_name_mapping['todo'] = 'Todo'
app_name_mapping['simply'] = 'Simply'
app_name_mapping['shopping'] = 'Shopping'

app_name_mapping['geek'] = 'Geek'
app_name_mapping['wish'] = 'Wish'
app_name_mapping['rainbow'] = 'Rainbow'
app_name_mapping['etsy'] = 'Etsy'
app_name_mapping['yelp'] = 'Yelp'

app_name_mapping['k9'] = 'K9'
app_name_mapping['email'] = 'Email'
app_name_mapping['ru'] = 'Ru'
app_name_mapping['mymail'] = 'Mymail'
app_name_mapping['any'] = 'Any'


app_name_mapping['tipcalculator'] = 'Tipcalculator'
app_name_mapping['tipcalc'] = 'Tipcalc'
app_name_mapping['simple'] = 'Simple'
app_name_mapping['tipplus'] = 'Tipplus'
app_name_mapping['free'] = 'Free'


def append_src_gt_events(test):
    events = {}
    # add src events
    src_events = ground_truth_tests.loc[ground_truth_tests['method'] == test['method']]
    if src_events.shape[0] == 1:
        events['src_events'] = [gui_event['id_or_xpath'] for gui_event in src_events.iloc[0]['event_array']]
    else:
        print('src events len is not 1, check: ', test['method'])
    # add gt events, need to use app_name_mapping 
    # otherwise line 'target_method = test['method'].replace(source_app, target_app)' will not replace
    source_app = app_name_mapping[test['source']]
    target_app = app_name_mapping[test['target']]
    # source_app = test['source']
    # target_app = test['target']
    target_method = test['method'].replace(source_app, target_app)
    gt_test = ground_truth_tests.loc[ground_truth_tests['method'] == target_method]
    if gt_test.shape[0] == 1:
        events['gt_events'] = [gui_event['id_or_xpath'] for gui_event in gt_test.iloc[0]['event_array']]
    else:
        if gt_test.shape[0] > 1:
            print('gt events len > 1, check: ', target_method)    
            print(gt_test.shape)
            print(gt_test)
    return events

def count_event_num_other(test):
    num_events = {}
    num_events['num_src'] = len(eval(test['src_events']))
    transferred_json = json.loads(test['event_array'])
    count = 0
    for trans in transferred_json:
        if trans['id_or_xpath'] != 'NONE':
            count += 1
    num_events['num_trans'] = count
    if pd.isnull(test['gt_events']):
        num_events['num_gt'] = np.NaN
    else:
        num_events['num_gt'] = len(eval(test['gt_events']))
    return num_events


def calculate_utility_other(test):
    result = {}
    try:
        result['reduction'] = (test['num_gt'] - test['distance']) / test['num_gt']
    except ZeroDivisionError:
        result['reduction'] = np.NaN
    return result

# using CraftDroid as an example
craftdroid_csv = []
# for path in glob.glob("input1/treadroid/mapping_results/*.csv"):
for path in glob.glob("input1/b4/*.csv"):
    csv = read_csv(path)
    apps = os.path.splitext(os.path.basename(path))[0].split("_")

    csv['source'] = csv.apply(lambda x: apps[0], axis=1)
    csv['target'] = csv.apply(lambda x: apps[1], axis=1)
    print("source", csv['source'])
    print("target",csv['target'])
    csv['gui_mapper'] = csv.apply(lambda x: "atm", axis=1)

    craftdroid_csv.append(csv)
    # print("craftdroid_csv",craftdroid_csv)
combined_csv = pd.concat(craftdroid_csv)
print("combined_csv",combined_csv)
combined_csv['event_array'] = combined_csv['event_array'].apply(json.loads)

ground_truth_tests = [read_csv(path, header=0) for path in glob.glob("input1/extracted_tests/*.csv")]
ground_truth_tests = pd.concat(ground_truth_tests)
ground_truth_tests['event_array'] = ground_truth_tests['event_array'].apply(json.loads)

combined_csv = pd.concat([combined_csv, combined_csv.apply(append_src_gt_events, axis=1).apply(pd.Series)], axis=1)
combined_csv['event_array'] = combined_csv['event_array'].apply(json.dumps)
combined_csv = pd.concat([combined_csv, combined_csv.apply(levenshtein, axis=1).apply(pd.Series)], axis=1)

combined_csv.to_csv("tmp.csv", index=False)
combined_csv = read_csv("tmp.csv")
combined_csv = pd.concat([combined_csv, combined_csv.apply(count_event_num_other, axis=1).apply(pd.Series)], axis=1)
combined_csv = pd.concat([combined_csv, combined_csv.apply(calculate_utility_other, axis=1).apply(pd.Series)], axis=1)
combined_csv
combined_csv.to_csv("input1/b4_utility.csv", index=False)
print('Done! Check the output file in /output1/b4_utility.csv')


# In[ ]:




