from mimetypes import init
from operator import length_hint
from tabnanny import check
import time
from copy import deepcopy
import sys
import traceback
import os
from statistics import mean
import pickle
from collections import defaultdict
import math
from datetime import datetime
from tracemalloc import start
from unittest import result
import numpy as np
import heapq

# local import
from Util import Util
from StrUtil import StrUtil
from Configuration import Configuration
from Runner import Runner
from WidgetUtil import WidgetUtil
# from misc import teardown_mail
from CallGraphParser import CallGraphParser
from ResourceParser import ResourceParser
from const import SA_INFO_FOLDER, SNAPSHOT_FOLDER


class Explorer:
    def __init__(self, config_id, appium_port='4723', udid=None):
        self.config = Configuration(config_id)
        self.runner = Runner(self.config.pkg_to, self.config.act_to, self.config.no_reset, appium_port, udid)
        self.src_events = Util.load_events(self.config.id, 'base_from')
        self.tid = self.config.id
        self.current_src_index = 0
        self.tgt_events = []
        self.f_target = 0
        self.prev_tgt_events = []
        self.f_prev_target = -1
        # self.is_rerun_required = True
        self.rp = ResourceParser(os.path.join(SA_INFO_FOLDER, self.config.id.split('-')[1]))
        self.widget_db = self.generate_widget_db()
        self.cgp = CallGraphParser(os.path.join(SA_INFO_FOLDER, self.config.id.split('-')[1]))
        self.invalid_events = defaultdict(list)
        self.nearest_button_to_text = None
        self.idx_src_to_tgt = {}
        self.skipped_match = defaultdict(list)
        self.consider_naf_only_widget = False
        self.steppings={}
        self.activitypath={}
        

    def generate_widget_db(self):
        db = {}
        for w in self.rp.get_widgets():
            if w['activity']:
                # the signature here has no 'clickable' and 'password', bcuz it's from static info
                w_signature = WidgetUtil.get_widget_signature(w)
                db[w_signature] = w
        return db

    def mutate_src_action(self, mutant):
        # e.g., mutant = {'long_press': 'swipe_right', 'swipe_right': 'long_press'}
        for e in self.src_events:
            if e['action'][0] in mutant:
                e['action'][0] = mutant[e['action'][0]]
        return self.src_events

    def run(self):
        # todo: or exceed a time limit
        # while self.f_target - self.f_prev_target > 0.001:  # still found a better solution
            # print('--\nStart a new round to find a better tgt event sequence')
            # print('Timestamp:', datetime.now())
        while self.f_target<= 0.12 and self.f_target - self.f_prev_target > 0.001:
            
            self.f_prev_target = self.f_target
            self.prev_tgt_events = self.tgt_events
            self.tgt_events = []
            self.tgt_eventslist = []
            self.current_src_index = 0


            self.invalid_events = defaultdict(list)
            self.skipped_match = defaultdict(list)
            self.idx_src_to_tgt = {}
            is_explored = False
            
            
            
            self.tgt_events=self.gen_tarevents(is_explored)

            self.f_target = self.fitness(self.tgt_events)
            print(f'Current target events with fitness {self.f_target}:')



            if self.f_target == self.f_prev_target == 0:
                self.reset_and_explore()
                self.tgt_events = []
                self.tgt_eventslist = []
                self.prev_tgt_events = []
                self.f_prev_target = -1
                continue

            if self.tgt_events[0]['class'] == 'EMPTY_EVENT':
                if 'stepping_events' not in self.src_events[1].keys() or self.src_events[1]['stepping_events']==[]:
                    print(' Explore the app and start over.')
                    self.reset_and_explore()
                    self.tgt_events = []
                    self.tgt_eventslist = []
                    self.prev_tgt_events = []
                    continue


            if self.current_src_index<len(self.src_events):
                tgt_events=self.tgt_events
                self.reset_and_explore(tgt_events)
                self.tgt_events = []
                self.tgt_eventslist = []
                self.prev_tgt_events = []
                self.f_prev_target = -1
                self.f_target =0.001
                continue


            print("self.tgt_events",self.tgt_events)
            if self.src_events[0]['event_type']=="stepping":
                if self.tgt_events[0]['class']=="EMPTY_EVENT" and self.tgt_events[1]['class']=="EMPTY_EVENT" and self.tgt_events[2]['event_type']=="stepping":
                    self.f_target = 0.1
            else:
                if self.tgt_events[0]['class']=="EMPTY_EVENT" and self.tgt_events[1]['event_type']=="stepping":
                    self.f_target = 0.1
            oraclenum=0
            tgt_oraclenum=0
            for i in self.src_events:
                if i['event_type']=='oracle':
                    oraclenum+=1
            if oraclenum>2:
                for i in self.tgt_events:
                    if i['event_type']=='oracle':
                        if i['class']=="EMPTY_EVENT":
                            tgt_oraclenum+=1
                if tgt_oraclenum>=2 and self.tgt_events[-1]['class']=='EMPTY_EVENT':
                    self.f_target = 0.1


        return True, 0

    def adj(self,results):
        resultsini=self.tgt_eventslist
        print("resultsini",resultsini)
        print("results",)
        s_lenth=len(self.src_events)
        t_lenth=len(resultsini)
        t_s =np.zeros([t_lenth, s_lenth])
        adjrange=int((s_lenth+1)/2)
        results_adj=self.tgt_events
        # [0,1,2,3]  2   [0,1,2,3,4] 3
        # print("src_events",self.src_events)
        # resultsini [[{'class': 'android.widget.EditText', 'resource-id': 'bill', 'id-prefix': 'anti.tip:id/', 'text': '0.00', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'naf': '', 'parent_text': '', 'sibling_text': ' Bill Amount', 'stepping_events': [], 'package': 'anti.tip', 'activity': '.Tip', 'event_type': 'gui', 'score': 0.3437140043824911, 'action': ['clear_and_send_keys', '56.6']}], [{'class': 'android.widget.EditText', 'resource-id': 'percent', 'id-prefix': 'anti.tip:id/', 'text': '15', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'naf': '', 'parent_text': '', 'sibling_text': ' Tip Percent (%)', 'stepping_events': [], 'package': 'anti.tip', 'activity': '.Tip', 'event_type': 'gui', 'score': 0.2718383527050416, 'action': ['clear_and_send_keys_and_hide_keyboard', '15']}], [{'class': 'android.widget.EditText', 'resource-id': 'each', 'id-prefix': 'anti.tip:id/', 'text': '65.09', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'naf': '', 'parent_text': '', 'sibling_text': ' Each Person Pays', 'stepping_events': [], 'package': 'anti.tip', 'activity': '.Tip', 'event_type': 'gui', 'score': 0.07665948507686456, 'action': ['clear_and_send_keys_and_hide_keyboard', '4']}], [{'class': 'EMPTY_EVENT', 'score': 0, 'event_type': 'oracle'}]]
        # src_events [{'class': 'android.widget.EditText', 'resource-id': 'com.appsbyvir.tipcalculator:id/billEditText', 'text': '', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'parent_text': '', 'sibling_text': 'Bill', 'tid': 'a52', 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'ignorable': 'false', 'event_type': 'gui', 'action': ['clear_and_send_keys', '56.6']}, {'class': 'android.widget.EditText', 'resource-id': 'com.appsbyvir.tipcalculator:id/tipEditText', 'text': '15.0', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'parent_text': '', 'sibling_text': 'Tip %', 'tid': 'a52', 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'ignorable': 'false', 'event_type': 'gui', 'action': ['clear_and_send_keys_and_hide_keyboard', '15']}, {'class': 'android.widget.EditText', 'resource-id': 'com.appsbyvir.tipcalculator:id/splitBillEditText', 'text': '# of ways', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'parent_text': '', 'sibling_text': 'Split Bill', 'tid': 'a52', 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'ignorable': 'false', 'event_type': 'gui', 'action': ['clear_and_send_keys_and_hide_keyboard', '4']}, {'class': 'android.widget.EditText', 'resource-id': 'com.appsbyvir.tipcalculator:id/perPersonEditText', 'text': '16.27', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'parent_text': '', 'sibling_text': 'Per Person:', 'tid': 'a52', 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'ignorable': 'false', 'event_type': 'oracle', 'action': ['wait_until_text_presence', 100, 'text', '16.27']}]

        # 初始化并更新矩阵

        # results_adj [{'class': 'android.widget.EditText', 'resource-id': 'perPersonEditText', 'id-prefix': 'com.appsbyvir.tipcalculator:id/', 'text': '0.0', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'naf': '', 'parent_text': '', 'sibling_text': 'Per Person:', 'stepping_events': [], 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'event_type': 'gui', 'score': 0.3465997762978077, 'action': ['clear_and_send_keys', '56.6']}, {'class': 'android.widget.EditText', 'resource-id': 'tipEditText', 'id-prefix': 'com.appsbyvir.tipcalculator:id/', 'text': '15.0', 'content-desc': '', 'clickable': 'true', 'password': 'false', 'naf': '', 'parent_text': '', 'sibling_text': 'Tip %', 'stepping_events': [], 'package': 'com.appsbyvir.tipcalculator', 'activity': '.MainActivity', 'event_type': 'gui', 'score': 0.5833333333333334, 'action': ['clear_and_send_keys_and_hide_keyboard', '15']}, {'class': 'EMPTY_EVENT', 'score': 0, 'event_type': 'gui'}, {'class': 'EMPTY_EVENT', 'score': 0, 'event_type': 'oracle'}]

        # if 'stepping_events' in list(explorer.src_events[1].keys()) \
        #     and self.src_events[1]['stepping_events'] != [] and results[0]['class']!='EMPTY_EVENT' and self.tgt_events[1]['stepping_events']==[]:
        #         print("explore-----------")
        #         scorelist = []
        #         for i in range(0, s_lenth):
        #             score1 = resultsini[i][-1]['score']
        #             scorelist.append(score1)
        #         print("scorelist", scorelist)
        #         te=0
        #         if scorelist.index(max(scorelist[0:3])) in [1,2]:
        #             results[0] = {"class": "EMPTY_EVENT", "score": 0, "event_type": "gui"}
        #             self.tgt_events[0]=results[0]
        #             self.tgt_eventslist[0]=[results[0]]
        #             te=1
        #         # else:
        #         #
        #         #     if scorelist.index(max(scorelist[0:3])) == 2:
        #         #         for i in range(3):
        #         #             results[i]={"class": "EMPTY_EVENT", "score": 0, "event_type": "gui"}
        #         #             self.tgt_events[i] = results[i]
        #         #             self.tgt_eventslist[i]=[results[i]]
        #         #         te=2
        #         print("te",te)
        #         if te!=0:
        #             self.f_prev_target = self.f_target
        #             self.prev_tgt_events = self.tgt_events
        #             self.prev_tgt_eventslist = self.tgt_eventslist
        #             self.current_src_index = te
        #             is_explored = False
        #             self.tgt_events = self.tgt_events[0:te]
        #             self.tgt_eventslist = self.tgt_eventslist[0:te]
        #             adj0={}
        #
        #             adjri = None
        #
        #             # if self.current_src_index in list(adj0.keys()):
        #             #     adj0[self.current_src_index].append(adjri)
        #             # else:
        #             #     adj0[self.current_src_index] = [adjri]
        #             # print("adj0", adj0)
        #             self.tgt_events = self.gen_tarevents_adj(is_explored, adjri, adj0)
        #             if self.tgt_events[-1]['class']!="EMPTY_EVENT" or self.f_target >= self.f_prev_target:
        #                 results_adj=self.tgt_events
        # else:
        for i in range(0, s_lenth):

            t_s[i][i] = resultsini[i][-1]['score']
            w = resultsini[i][-1]

            start = i + 1
            end = i + adjrange + 1
            if w['class'] == 'EMPTY_EVENT' or w['class'] == 'SYS_EVENT':
                continue
            src_valid = self.src_events[start:end]
            w_res_id = w['resource-id']
            print(src_valid, "src_valid", w_res_id, "w_res_id")
            for m in src_valid:
                if m['class'] == 'SYS_EVENT':
                    src_valid.remove(m)
                    continue
                if m['resource-id'] == w_res_id:
                    src_valid.remove(m)

            src_valid_candidates = WidgetUtil.most_similar(w, src_valid, self.config.use_stopwords,
                                                           self.config.expand_btn_to_text,
                                                           self.config.cross_check)
            print("src_valid_candidates", src_valid_candidates)

            for v, (src, _) in enumerate(src_valid_candidates):
                if _ > t_s[i][i]:
                    j = self.src_events.index(src)
                    t_s[i][j] = _
            print("t_s", t_s)
            tindex = []
            toindex = []

            for i in range(0, s_lenth):
                j = i + 1
                while j < s_lenth:
                    if t_s[i][j] > t_s[i][i]:
                        tindex.append(i)
                        toindex.append(j)
                    j = j + 1
            print("tindex,toindex", tindex, toindex)
            # tindex,toindex [2, 2, 3, 3, 6, 6, 6] [4, 5, 4, 5, 7, 8, 9]
            if len(tindex) > 3:
                tindex = []

            if tindex != []:
                self.f_prev_target = self.f_target
                self.prev_tgt_events = self.tgt_events
                self.tgt_events = self.tgt_events[0:tindex[0]]
                self.current_src_index = tindex[0]
                self.tgt_eventslist = self.tgt_eventslist[0:tindex[0]]

                is_explored = False
                adjri = resultsini[self.current_src_index][-1]
                adj0 = {}
                self.tgt_events = self.gen_tarevents_adj(is_explored, adjri, adj0)
                self.f_target = self.fitness(self.tgt_events)
                print(f'Current target events with fitness {self.f_target}:')
                if self.f_target >= self.f_prev_target:
                    results_adj = self.tgt_events
                else:
                    results_adj = results
                    self.tgt_eventslist = resultsini
                    self.f_target = self.f_prev_target
                    self.tgt_events = self.prev_tgt_events

                    print("self.f_prev_target", self.f_prev_target)
                    print("self.f_target", self.f_target)
            else:
                results_adj = results
            print("results_adj", results_adj)
            print("self.f_prev_target", self.f_prev_target)



        if results_adj[-1]['class']=="EMPTY_EVENT" or results_adj[-1]['action'][0]=="wait_until_element_presence" and self.src_events[-1]['action'][-1]!=results_adj[-1]['action'][-1]:
            # and self.src_events[1]['stepping_events']== []:

            resultsini=self.tgt_eventslist
            temp=self.get_index(resultsini)

            if temp!=[]:
                v=0
                while v<2:
                    self.f_prev_target = self.f_target
                    self.prev_tgt_events = self.tgt_events
                    self.prev_tgt_eventslist = self.tgt_eventslist
                    check=1
                    adj0={}
                    while check<=2:
                        print("check",check)
                        self.current_src_index = temp[v]

                        is_explored = False
                        print("self.tgt_eventslist",self.tgt_eventslist)

                        adjri=self.tgt_eventslist[self.current_src_index][-1]

                        if self.current_src_index in list(adj0.keys()):
                            adj0[self.current_src_index].append(adjri)
                        else:
                            adj0[self.current_src_index]=[adjri]
                        print("adj0",adj0)
                        for i in adj0[self.current_src_index]:
                            if i['class']=="EMPTY_EVENT":
                                continue
                        if temp[v]==0:
                            self.tgt_events = []
                            self.tgt_eventslist=[]
                        else:
                            self.tgt_events = self.tgt_events[0:temp[v]]
                            self.tgt_eventslist=self.tgt_eventslist[0:temp[v]]


                        print("self.current_src_index",self.current_src_index)


                        self.tgt_events=self.gen_tarevents_adj(is_explored,adjri,adj0)
                        self.f_target = self.fitness(self.tgt_events)
                        print(f'Current target events with fitness {self.f_target}:')
                        print("self.f_target,self.f_prev_target",self.f_target,self.f_prev_target)
                        print("self.prev_tgt_eventslist,self.tgt_eventslist",self.prev_tgt_eventslist,self.tgt_eventslist)

                        if self.f_target <=self.f_prev_target:
                            if self.tgt_events[-1]['class']!='EMPTY_EVENT' and self.src_events[-1]['action'][-1]==self.tgt_events[-1]['action'][-1]:
                                break
                            check=check+1
                            adj0[temp[v]].append(self.tgt_eventslist[temp[v]][-1])

                            self.tgt_events =self.prev_tgt_events
                            self.f_target=self.f_prev_target
                            self.tgt_eventslist=self.prev_tgt_eventslist

                            # else:
                            #     self.tgt_events =self.prev_tgt_events
                            #     self.f_target=self.f_prev_target
                            #     self.tgt_eventslist=self.prev_tgt_eventslist
                            #     break
                        else:
                            break


                    if self.f_target >self.f_prev_target or self.tgt_events[-1]['class']!='EMPTY_EVENT' and self.src_events[-1]['action'][-1]==self.tgt_events[-1]['action'][-1]:
                        if self.tgt_events[-1]['class']=='EMPTY_EVENT':
                            resultsini = self.tgt_eventslist
                            temp = self.get_index(resultsini)
                            v=0
                        else:
                            break

                    else:
                        if self.tgt_events[-1]['class']!='EMPTY_EVENT' and self.src_events[-1]['action'][-1]==self.tgt_events[-1]['action'][-1]:
                            break

                        v=v+1
                        print("v",v)
                        self.tgt_events =self.prev_tgt_events
                        self.f_target=self.f_prev_target
                        self.tgt_eventslist=self.prev_tgt_eventslist
                        print("self.tgt_events",self.tgt_events)
                        print("self.tgt_eventslist",self.tgt_eventslist)


                results_final=self.tgt_events

            print("results_final",results_final)
        else:
            results_final=results_adj

            # adaptive0
        return results_final

    def get_index(self,resultsini):
        scorelist = []
        for i in range(0, len(self.src_events) - 1):
            score1 = resultsini[i][-1]['score']
            # if resultsini[i][0]['event_type']=='stepping':
            #     score1=resultsini[i][1]['score']
            # else:
            #     score1=resultsini[i][0]['score']
            if score1 == 0:
                scorelist.append(1.1)
            else:
                scorelist.append(score1)
        print("scorelist", scorelist)
        # 找出相似度最小的两个索引
        min_n = 2
        cs = 0
        scor11 = 0
        for scor in scorelist:
            if scor < 0.09:
                cs += 1
            if scor == 1.1:
                scor11 += 1
        if cs != 0:
            if cs >= 3 or scor11 > 4:
                for scorindex in range(0, len(scorelist)):
                    if scorelist[scorindex] < 0.09:
                        temp = [scorindex, scorindex - 1]
                        break


        # 记录连续小于0.05的索引，如果连续的超过3，则需要重新查找第一个小于0.05的索引以及前一个
        else:
            temp = map(scorelist.index, heapq.nsmallest(min_n, scorelist))
            temp = list(temp)
        print("temp", temp)
        if temp[0] == temp[1]:
            min_value = min(scorelist)
            temp = [i for i, x in enumerate(scorelist) if x == min_value]
        print("temp", temp)
        # [3, 2]
        return temp

    def gen_tarevents(self,is_explored):
        # Backtrackco=0
        ci=0
        csi0 = 0
        while self.current_src_index < len(self.src_events):
            src_event = self.src_events[self.current_src_index]
            print(f'Source Event:\n{src_event}')

            if self.current_src_index == len(self.src_events) - 1 and src_event['event_type'] == "oracle":
                self.consider_naf_only_widget = True  # e.g., a32-a33-b31

            else:
                self.consider_naf_only_widget = False  # e.g., a35-a33-b31

            tgt_event = None
            # Just replicate previous src_event if that is an oracle and current one is the action for it
            if self.current_src_index > 0:
                prev_src_event = self.src_events[self.current_src_index - 1]
                if prev_src_event['event_type'] == 'oracle' and src_event['event_type'] == 'gui' \
                        and WidgetUtil.is_equal(prev_src_event, src_event) \
                        and self.tgt_events[-1]['class'] != 'EMPTY_EVENT'\
                        and src_event['action'][0]!='send_keys'\
                        and src_event['action'][0]!='wait_until_element_presence':
                    tgt_event = deepcopy(self.tgt_events[-1])
                    if 'stepping_events' in tgt_event:
                        tgt_event['stepping_events'] = []
                    tgt_event['event_type'] = 'gui'
                    tgt_event['action'] = deepcopy(src_event['action'])
                    if self.check_skipped(tgt_event):  # don't copy previous src_event if it should be skipped
                        tgt_event = None
                    # todo: remember if the tgt_action is propagated the previous oracle;
                    #       if yes and skipped/rematched, also change the previous oracle

            if src_event['event_type'] == 'SYS_EVENT':
                # e.g., {"class": "SYS_EVENT","action": ["KEY_BACK"], "event_type":  "SYS_EVENT"}
                # self.is_rerun_required = False
                tgt_event = deepcopy(src_event)

            if src_event['event_type'] == "stepping":
                tgt_event = self.generate_empty_event('gui')



            backtrack = False

            if not tgt_event:
                try:
                    dom, pkg, act = self.execute_target_events([])
                    # print("act",act)
                except:  # selenium.common.exceptions.NoSuchElementException
                    # a23-a21-b21, a24-a21-b21: selected an EditText which is not editable
                    print(f'Backtrack to the previous step due to an exception in execution.')
                    invalid_event = self.tgt_events[-1]

                    if ci==0:

                        csi0=self.current_src_index
                        print("csi0",csi0)
                    if self.current_src_index==csi0:
                        ci=ci+1
                        print("ci",ci)

                    
                   
                    self.current_src_index -= 1
                        # pop tgt_events
                    if self.current_src_index == 0:
                        self.tgt_events = []
                    else:
                        self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[self.current_src_index - 1] + 1]
                    self.invalid_events[self.current_src_index].append(deepcopy(invalid_event))
                    print("self.invalid_events",self.invalid_events)
                    if ci >= 5:
                        break
                    continue



                self.cache_seen_widgets(dom, pkg, act)
                print("self.widget_db.values()",self.widget_db.values())

                w_candidates = []
                num_to_check = 10
                # print("self.widget_db.values()",self.widget_db.values())
                if src_event['action'][0] == 'wait_until_text_invisible':
                    if not self.nearest_button_to_text:
                        tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                    else:
                        w_candidates = WidgetUtil.most_similar(self.nearest_button_to_text, self.widget_db.values(),
                                                                self.config.use_stopwords,
                                                                self.config.expand_btn_to_text,
                                                                self.config.cross_check)
                        num_to_check = 1  # we know the button exists, so no need to seek other similar ones
                else:
                    w_candidates = WidgetUtil.most_similar(src_event, self.widget_db.values(),
                                                            self.config.use_stopwords,
                                                            self.config.expand_btn_to_text,
                                                            self.config.cross_check)

                # if w_candidates:
                #     w_candidates = self.decay_by_distance(w_candidates, pkg, act)

                print("w_candidates",w_candidates)
                

                if self.current_src_index > 0:
                    prev_src_event = self.src_events[self.current_src_index - 1]
                    if w_candidates==[] and prev_src_event['event_type'] == 'oracle':
                        st_event = deepcopy(self.tgt_events[-1])
                        if 'stepping_events' in st_event:
                            st_event['stepping_events'] = []
                        st_event['event_type'] = 'stepping'
                        st_event['action'] = ['click']
                        print("st_event",st_event)
                        dom, pkg, act = self.execute_target_events([st_event])
                        self.cache_seen_widgets(dom, pkg, act)
                        w_candidates = WidgetUtil.most_similar(src_event, self.widget_db.values(),
                                                               self.config.use_stopwords,
                                                               self.config.expand_btn_to_text,
                                                               self.config.cross_check)
                        print("w_candidates", w_candidates)
                        if w_candidates!=[]:
                            print(f'Backtrack to the previous step due to an additional exploration in execution.')
                            self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[self.current_src_index - 1]]
                            self.tgt_eventslist = self.tgt_eventslist[:self.idx_src_to_tgt[self.current_src_index - 1]]
                            self.current_src_index -= 1
                            continue
                
                for i, (w, _) in enumerate(w_candidates[:num_to_check]):
                    # encode-decode: for some weird chars in a1 apps
                    print(f'({i+1}/{num_to_check}) Validating Similar w: {w}'.encode("utf-8").decode("utf-8"))
                    # skip invalid events
                    if any([WidgetUtil.is_equal(w, e) for e in self.invalid_events.get(self.current_src_index, [])]):
                        print('Skip a known broken event:', w)
                        continue
                    if self.current_src_index==0:
                        if w['text'] in ['Skip','Next']:
                            continue

                    # skip widget with empty attribute if the action is wait_until with the attribute; a33-a35-b31
                    if src_event['action'][0] == 'wait_until_element_presence':
                        is_empty_atc = False
                        attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password', 'naf'})
                        for atc in attrs_to_check:
                            if not w[atc]:
                                atc_in_oracle = 'id' if atc == 'resource-id' else atc
                                if src_event['action'][2] == atc_in_oracle:
                                    is_empty_atc = True
                                    break
                                # a31-a33-b31; 'action': ['wait_until_element_presence', 10, 'xpath',
                                #                         '//android.widget.TextView[@content-desc=""]']
                                elif src_event['action'][2] == 'xpath' and '@'+atc in src_event['action'][3]:
                                    is_empty_atc = True
                                    break
                        if is_empty_atc:
                            print('Skip the widget without the attribute that the action is waiting for')
                            continue
                    try:
                        match = self.check_reachability(w, pkg, act)
                    except Exception as excep:
                        print(excep)
                        traceback.print_exc()
                        return False, self.current_src_index
                    if match:
                        # Never map two src EditText to the same tgt EditText, e.g., a51-a52-b52
                        if match['class'] == 'android.widget.EditText' and 'send_keys' in src_event['action'][0]:
                            if self.check_skipped(match):
                                print(f'Duplicated match (later): {match}\n. Skipped.')
                                continue
                            is_mapped, tgt_idx, src_idx = self.check_mapped(match)
                            # exact identical EditText in src_events, e.g., a12-a11-b12
                            is_idential_src_widgets = self.check_identical_src_widgets(src_idx, self.current_src_index)
                            if is_mapped and not is_idential_src_widgets:
                                if match['score'] <= self.tgt_events[tgt_idx]['score']:
                                    print(f'Duplicated match (previous): {match}\n. Skipped.')
                                    continue  # discard this match
                                else:
                                    print(f'Duplicated match. Backtrack to src_idx: {src_idx} to find another match')
                                    continue
                                    backtrack = True
                                    self.current_src_index = src_idx
                                    self.skipped_match[src_idx].append(deepcopy(self.tgt_events[tgt_idx]))
                                    # pop tgt_events
                                    if src_idx == 0:
                                        self.tgt_events = []
                                        self.tgt_eventslist=[]
                                    else:
                                        self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[src_idx-1] + 1]
                                    break
                        if 'clickable' not in w:  # a static widget
                            self.widget_db.pop(WidgetUtil.get_widget_signature(w), None)
                        if src_event['action'][0] == 'wait_until_text_invisible':
                            if self.runner.check_text_invisible(src_event):
                                tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                            else:
                                tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                        else:
                            # if w['clickable']=='true' and w['class']=='android.widget.Button':
                            #     tgt_event = self.generate_event(match, 'click')
                            # else:
                            tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                        break
            if backtrack:
                continue

            if not tgt_event:
                tgt_event = Explorer.generate_empty_event(src_event['event_type'])

            # additional exploration (ATG and widget_db update) for empty oracle (e.g., a51-a53-b51)
            if tgt_event['class'] == 'EMPTY_EVENT' and tgt_event['event_type'] == 'oracle' and not is_explored:
                print('Empty event for an oracle. Try to explore the app')
                self.reset_and_explore(self.tgt_events)
                is_explored = True
                continue
            else:
                is_explored = False

            print('** Learned for this step:')
            if 'stepping_events' in tgt_event and tgt_event['stepping_events']:
                self.tgt_events += tgt_event['stepping_events']
                # for t in tgt_event['stepping_events']:
                #     print(t)
            print(tgt_event)
            print('--')
            self.tgt_events.append(tgt_event)

            print("self.tgt_events", self.tgt_events)
            
            if 'stepping_events' in tgt_event and tgt_event['stepping_events']:
                self.tgt_eventslist.append(tgt_event['stepping_events']+[tgt_event])
            else:
                self.tgt_eventslist.append([tgt_event])
            self.idx_src_to_tgt[self.current_src_index] = len(self.tgt_events) - 1
            self.current_src_index += 1 
            print("self.tgt_eventslist",self.tgt_eventslist)
        return self.tgt_events


    def gen_tarevents_adj(self,is_explored,adjri,adj0):

        while self.current_src_index < len(self.src_events):
            src_event = self.src_events[self.current_src_index]
            print(f'Source Event:\n{src_event}')

            if self.current_src_index == len(self.src_events) - 1 and src_event['event_type'] == "oracle":
                self.consider_naf_only_widget = True  # e.g., a32-a33-b31

            else:
                self.consider_naf_only_widget = False  # e.g., a35-a33-b31

            tgt_event = None
            # Just replicate previous src_event if that is an oracle and current one is the action for it
            if self.current_src_index > 0:
                prev_src_event = self.src_events[self.current_src_index - 1]
                if prev_src_event['event_type'] == 'oracle' and src_event['event_type'] == 'gui' \
                        and WidgetUtil.is_equal(prev_src_event, src_event) \
                        and self.tgt_events[-1]['class'] != 'EMPTY_EVENT':
                    tgt_event = deepcopy(self.tgt_events[-1])
                    if 'stepping_events' in tgt_event:
                        tgt_event['stepping_events'] = []
                    tgt_event['event_type'] = 'gui'
                    tgt_event['action'] = deepcopy(src_event['action'])
                    if self.check_skipped(tgt_event):  # don't copy previous src_event if it should be skipped
                        tgt_event = None
                    # todo: remember if the tgt_action is propagated the previous oracle;
                    #       if yes and skipped/rematched, also change the previous oracle

            if src_event['event_type'] == 'SYS_EVENT':

                tgt_event = deepcopy(src_event)

            if src_event['event_type'] == "stepping":
                tgt_event = self.generate_empty_event('gui')

            backtrack = False
            if not tgt_event:
                try:
                    dom, pkg, act = self.execute_target_events([])
                except:  # selenium.common.exceptions.NoSuchElementException
                    # a23-a21-b21, a24-a21-b21: selected an EditText which is not editable
                    # print(f'Backtrack to the previous step due to an exception in execution.')
                    invalid_event = self.tgt_events[-1]
            
                    self.current_src_index -= 1
                    # pop tgt_events
                    if self.current_src_index == 0:
                        self.tgt_events = []
                    else:
                        self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[self.current_src_index - 1] + 1]
                    self.invalid_events[self.current_src_index].append(deepcopy(invalid_event))
                    continue

                self.cache_seen_widgets(dom, pkg, act)

                w_candidates = []
                num_to_check = 10
                # print("self.widget_db.values()",self.widget_db.values())
                if src_event['action'][0] == 'wait_until_text_invisible':
                    if not self.nearest_button_to_text:
                        tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                    else:
                        w_candidates = WidgetUtil.most_similar(self.nearest_button_to_text, self.widget_db.values(),
                                                                self.config.use_stopwords,
                                                                self.config.expand_btn_to_text,
                                                                self.config.cross_check)
                        num_to_check = 1  # we know the button exists, so no need to seek other similar ones
                else:
                    w_candidates = WidgetUtil.most_similar(src_event, self.widget_db.values(),
                                                            self.config.use_stopwords,
                                                            self.config.expand_btn_to_text,
                                                            self.config.cross_check)

                # if w_candidates:
                #     w_candidates = self.decay_by_distance(w_candidates, pkg, act)

                
                print("w_candidates",w_candidates)



                istart=0
                print("adjri",adjri)
                if adjri!=None:
                    if self.current_src_index!=0 and self.tgt_events[0]['class']!='EMPTY_EVENT':
                        for i, (w, _) in enumerate(w_candidates[:num_to_check]):

                            if w['resource-id'] ==adjri['resource-id']:
                                i0=i
                                istart=i0+1
                                break
                print("istart,adjri",istart,adjri)
                for i, (w, _) in enumerate(w_candidates[istart:num_to_check]):
                    if adjri!=None and adjri['class']!='EMPTY_EVENT':
                        # self.tgt_events[0]['class']!='EMPTY_EVENT'
                        if w['resource-id']==adjri['resource-id']:
                            continue
                    b=0
                    if self.current_src_index in list(adj0.keys()):

                        for a in adj0[self.current_src_index]:
                            if adjri['class']!='EMPTY_EVENT' and w['resource-id']==a['resource-id']:
                                b=b+1
                                break
                    if b!=0:
                        continue

                    # encode-decode: for some weird chars in a1 apps
                    print(f'({i+1}/{num_to_check}) Validating Similar w: {w}'.encode("utf-8").decode("utf-8"))
                    # skip invalid events
                    # print("w",w)
                    if any([WidgetUtil.is_equal(w, e) for e in self.invalid_events.get(self.current_src_index, [])]):
                        # print('Skip a known broken event:', w)
                        continue

                    if self.current_src_index==0:
                        if w['text'] in ['Skip','Next']:
                            continue
                    # skip widget with empty attribute if the action is wait_until with the attribute; a33-a35-b31
                    if src_event['action'][0] == 'wait_until_element_presence':
                        is_empty_atc = False
                        attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password', 'naf'})
                        for atc in attrs_to_check:
                            if not w[atc]:
                                atc_in_oracle = 'id' if atc == 'resource-id' else atc
                                if src_event['action'][2] == atc_in_oracle:
                                    is_empty_atc = True
                                    break
                                # a31-a33-b31; 'action': ['wait_until_element_presence', 10, 'xpath',
                                #                         '//android.widget.TextView[@content-desc=""]']
                                elif src_event['action'][2] == 'xpath' and '@'+atc in src_event['action'][3]:
                                    is_empty_atc = True
                                    break
                        if is_empty_atc:
                            print('Skip the widget without the attribute that the action is waiting for')
                            continue
                    try:
                        match = self.check_reachability(w, pkg, act)
                    except Exception as excep:
                        print(excep)
                        traceback.print_exc()
                        return False, self.current_src_index
                    if match:
                        # Never map two src EditText to the same tgt EditText, e.g., a51-a52-b52
                        if match['class'] == 'android.widget.EditText' and 'send_keys' in src_event['action'][0]:
                            if self.check_skipped(match):
                                print(f'Duplicated match (later): {match}\n. Skipped.')
                                continue
                            is_mapped, tgt_idx, src_idx = self.check_mapped(match)
                            # exact identical EditText in src_events, e.g., a12-a11-b12
                            is_idential_src_widgets = self.check_identical_src_widgets(src_idx, self.current_src_index)
                            if is_mapped and not is_idential_src_widgets:
                                if match['score'] <= self.tgt_events[tgt_idx]['score']:
                                    print(f'Duplicated match (previous): {match}\n. Skipped.')
                                    continue  # discard this match
                                else:
                                    print(f'Duplicated match. Backtrack to src_idx: {src_idx} to find another match')
                                    continue
                                    # backtrack = True
                                    # self.current_src_index = src_idx
                                    # self.skipped_match[src_idx].append(deepcopy(self.tgt_events[tgt_idx]))
                                    # # pop tgt_events
                                    # if src_idx == 0:
                                    #     self.tgt_events = []
                                    # else:
                                    #     self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[src_idx-1] + 1]
                                    # break
                        if 'clickable' not in w:  # a static widget
                            self.widget_db.pop(WidgetUtil.get_widget_signature(w), None)
                        if src_event['action'][0] == 'wait_until_text_invisible':
                            if self.runner.check_text_invisible(src_event):
                                tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                            else:
                                tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                        else:
                            tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                        break
            if backtrack:
                continue

            if not tgt_event:
                tgt_event = Explorer.generate_empty_event(src_event['event_type'])



            # if tgt_event['class'] == 'EMPTY_EVENT' and src_event['event_type'] != "stepping":
            #     break
            # 如果在探索过程中生成空事件，则终止后面以节省时间

            # additional exploration (ATG and widget_db update) for empty oracle (e.g., a51-a53-b51)
            if tgt_event['class'] == 'EMPTY_EVENT' and tgt_event['event_type'] == 'oracle' and not is_explored:
                # print('Empty event for an oracle. Try to explore the app')
                self.reset_and_explore(self.tgt_events)
                is_explored = True
                continue
            else:
                is_explored = False

            print('** Learned for this step:')
            if 'stepping_events' in tgt_event and tgt_event['stepping_events']:
                self.tgt_events += tgt_event['stepping_events']
                # for t in tgt_event['stepping_events']:
                #     print(t)
            print(self.tgt_events)
            print('--')
            self.tgt_events.append(tgt_event)
            if 'stepping_events' in tgt_event and tgt_event['stepping_events']:
                self.tgt_eventslist.append(tgt_event['stepping_events']+[tgt_event])
            else:
                self.tgt_eventslist.append([tgt_event])
            self.idx_src_to_tgt[self.current_src_index] = len(self.tgt_events) - 1
            self.current_src_index += 1 
            adjri=None
        return self.tgt_events


    def reset_and_explore(self, tgt_events=[]):
        """Reset current state to the one after executing tgt_events
           and update ATG and widget_db by systematic exploration
        """
        print("tgt_events",tgt_events)
        print(1)
        self.runner.perform_actions(tgt_events, reset=True)  # reset app
        all_widgets = WidgetUtil.find_all_widgets(self.runner.get_page_source(),
                                                  self.runner.get_current_package(),
                                                  self.runner.get_current_activity(),
                                                  self.config.pkg_to)
        btn_widgets = []
        for w in all_widgets:
            if w['class'] in ['android.widget.Button', 'android.widget.ImageButton', 'android.widget.TextView']:
                attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'class', 'clickable', 'password'})
                attr_check = [attr in w and w[attr] for attr in attrs_to_check]
                if w['clickable'] == 'true' and any(attr_check):
                    btn_widgets.append(w)
        for btn_w in btn_widgets:
            print(2)
            self.runner.perform_actions(tgt_events, reset=True)
            btn_w['action'] = ['click']
            self.runner.perform_actions([btn_w], reset=False, cgp=self.cgp)
            self.cache_seen_widgets(self.runner.get_page_source(),
                                    self.runner.get_current_package(),
                                    self.runner.get_current_activity())

    def cache_seen_widgets(self, dom, pkg, act):
        current_widgets = WidgetUtil.find_all_widgets(dom, pkg, act, self.config.pkg_to)
        # print("current_widgets",current_widgets)
        # print('** before:', self.widget_db)
        for w in current_widgets:
            w_signature = WidgetUtil.get_widget_signature(w)

            # remove the widget from sa info if already seen here
            w_sa = {k: v for k, v in w.items() if k not in ['clickable', 'password']}
            w_sa_signature = WidgetUtil.get_widget_signature(w_sa)
            popped = self.widget_db.pop(w_sa_signature, None)
            if popped:
                print('** wDB (SA) popped:', popped)

            # remove useless email fields with obsolete email address
            tmp_email = self.runner.databank.get_temp_email(renew=False)
            if tmp_email in w_signature:
                pre = w_signature.split(tmp_email)[0]
                if not pre.endswith('!'):
                    pre = pre.replace(pre.split('!')[-1], '', 1)
                post = w_signature.split(tmp_email)[-1]
                if not post.startswith('!'):
                    post = post.replace(post.split('!')[0], '', 1)
                discarded_keys = []
                for k in self.widget_db.keys():
                    if k.startswith(pre) and k.endswith(post) and k != pre + post:
                        if StrUtil.is_contain_email(self.widget_db[k]['text']):
                            discarded_keys.append(k)
                for k in discarded_keys:
                    popped = self.widget_db.pop(k, None)
                    # if popped:
                    #     print('** wDB (obsolete Email) popped:', popped)

            # print('** wDB updated:', w)
            self.widget_db[w_signature] = w
        # print('** after:', self.widget_db)

    def execute_target_events(self, stepping_events):
        src_event = self.src_events[self.current_src_index]
        require_wait = src_event['action'][0].startswith('wait_until')
        # require_wait = True
        # if self.is_rerun_required:
        self.runner.perform_actions(self.tgt_events, require_wait, reset=True, cgp=self.cgp)
        # elif not self.is_rerun_required and self.tgt_events:
        #     # no reset and rerun, just execute the last matched action
        #     self.runner.perform_actions([self.tgt_events[-1]], require_wait, reset=False, cgp=self.cgp)
        self.runner.perform_actions(stepping_events, require_wait, reset=False, cgp=self.cgp)

        return self.runner.get_page_source(), self.runner.get_current_package(), self.runner.get_current_activity()

    @staticmethod
    def generate_event(w, actions=None):
        # if the action is wait_until_presence, change the content-desc/text/id to that of target app
        # e.g., ['wait_until_element_presence', 10, 'xpath', '//*[@content-desc="Open Menu"]']
        if actions[0] == 'wait_until_element_presence':
            if actions[2] == 'xpath' and '@content-desc=' in actions[3]:
                pre, post = actions[3].split('@content-desc=')
                post = f'@content-desc="{w["content-desc"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'xpath' and '@text=' in actions[3]:
                pre, post = actions[3].split('@text=')
                post = f'@text="{w["text"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'xpath' and 'contains(@text,' in actions[3]:
                pre, post = actions[3].split('contains(@text,')
                post = f'contains(@text, "{w["text"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'id':
                actions[3] = w['resource-id']
        w['action'] = actions
        return w

    @staticmethod
    def generate_empty_event(event_type):
        return {"class": "EMPTY_EVENT", 'score': 0, 'event_type': event_type}

    def check_reachability(self, w, current_pkg, current_act):

        from1 = current_pkg + current_act
        to1 = w['package'] + w['activity']
        potential_paths = self.cgp.get_paths_between_activities(from1, to1, self.consider_naf_only_widget)
        start_stepping=0

        potential_paths.insert(0, [])
        # print(f'Activity transition: {current_act} -> {w_act}. {len(potential_paths)} paths to validate.')

        if self.current_src_index==0:
            for ppath in potential_paths:
                path_show=self.path_show(ppath)
                if path_show != [] and path_show[1].startswith('D@'):
                    ab_pairs = path_show[1][2:].split('&')
                    ab = [kvp.split('=') for kvp in ab_pairs]
                    for j in ab:
                        if j[0] in ['resource-id','text']:
                            if 'skip' in j or 'Skip' in j:
                                start_stepping=start_stepping+1
                if start_stepping>=1:
                    break
            if [] in potential_paths and start_stepping>=1:

                potential_paths.remove([])
            

        invalid_paths = []
        for ppath in potential_paths:
            match = self.validate_path(ppath, w, invalid_paths,current_act)
            print("match",match)
            if match:
                return match
        return None

    def validate_path(self, ppath, w_target, invalid_paths,current_act):
        path_show=self.path_show(ppath)
        # print(f'Validating path: ', path_show)
        if None in path_show:
            return None
        ppath0=''.join(path_show)
        w_act=w_target['activity']
        
        if current_act+w_act not in self.activitypath.keys():
            self.activitypath.setdefault(current_act+w_act,[])
        if ppath0 not in self.activitypath[current_act+w_act]:
            self.activitypath[current_act+w_act].append(ppath0)
            
            for ip in invalid_paths:
                if ip == ppath[:len(ip)]:
                    # print('Known invalid path prefix:', ppath[:len(ip)])
                    self.activitypath[current_act+w_act].remove(ppath0)
                    return None
        

        # start follow the path to w_target
            _, __, ___ = self.execute_target_events([])
            stepping = []
            for i, hop in enumerate(ppath):
                if '(' in hop:  # a GUI event
                    w_id = ' '.join(hop.split()[:-1])
                    action = hop.split('(')[1][:-1]
                    action = 'long_press' if action in ['onItemLongClick', 'onLongClick'] else 'click'
                    if w_id.startswith('D@'):  # from dynamic exploration
                        # e.g., 'D@class=android.widget.Button&resource-id=org.secuso.privacyfriendlytodolist:id/btn_skip&text=Skip&content-desc='
                        kv_pairs = w_id[2:].split('&')
                        kv = [kvp.split('=') for kvp in kv_pairs]
                        kv1=[]

                        for j in kv:
                            if j[0] not in ['resource-id','text','class','naf','content-desc']:
                                for m in ['naf','text','content-desc','resource-id','class']:
                                    for pairs in kv:
                                        if pairs[0]==m:
                                            kv1.append(pairs)
                                            continue
                                kv1.insert(2,j)
                                kv=kv1
            # ['douzifly.list.ui.home.MainActivity', 'D@content-desc=&text=&resource-id=fab_add&class=android.widget.ImageButton&naf=true', 'douzifly.list.ui.home.MainActivity']
                        # print("kv",kv)
                        for pairs in kv:
                            if len(pairs)==3:
                                pairs0=pairs[1]+'='+pairs[2]
                                kv[1]=[pairs[0],pairs0] 
                            if len(pairs) == 2:
                                if pairs[0] not in ['resource-id','text','class','naf','content-desc']:
                                    pairs0=pairs[0]+'='+pairs[1]
                                    kv.remove(pairs)
                                    kv[1][1]+='&'+pairs0
                                    

                        criteria = {k: v for k, v in kv}
                        w_stepping = WidgetUtil.locate_widget(self.runner.get_page_source(), criteria)
                    else:  # from static analysis
                        w_name = self.rp.get_wName_from_oId(w_id)
                        w_stepping = WidgetUtil.locate_widget(self.runner.get_page_source(), {'resource-id': w_name})
                    if not w_stepping:
                        is_existed = False
                        for ip in invalid_paths:
                            if ip == ppath[:i+1]:
                                is_existed = True
                        if not is_existed:
                            invalid_paths.append([h for h in ppath[:i+1]])
                        self.activitypath[current_act+w_act].remove(ppath0)
                        return None
                    w_stepping['action'] = [action]
                    w_stepping['activity'] = self.runner.get_current_activity()
                    w_stepping['package'] = self.runner.get_current_package()
                    w_stepping['event_type'] = 'stepping'
                    stepping.append(w_stepping)
                    print("stepping",stepping)
                    act_from = self.runner.get_current_package() + self.runner.get_current_activity()
                    self.runner.perform_actions([stepping[-1]], require_wait=False, reset=False, cgp=self.cgp)
                    self.cache_seen_widgets(self.runner.get_page_source(),
                                            self.runner.get_current_package(),
                                            self.runner.get_current_activity())
                    act_to = self.runner.get_current_package() + self.runner.get_current_activity()
                    self.cgp.add_edge(act_from, act_to, w_stepping)
            self.steppings.setdefault(ppath0, []).append(stepping)

            attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password'})
            criteria = {k: w_target[k] for k in attrs_to_check if k in w_target}

            if self.src_events[self.current_src_index]['action'][0] == 'wait_until_text_presence':
                criteria['text'] = self.src_events[self.current_src_index]['action'][3]
            if self.current_src_index > 0 and self.is_for_email_or_pwd(self.src_events[self.current_src_index-1],
                                                                    self.src_events[self.current_src_index]):
                if StrUtil.is_contain_email(self.src_events[self.current_src_index]['action'][1]):
                    criteria['text'] = self.runner.databank.get_temp_email(renew=False)
            w_tgt = WidgetUtil.locate_widget(self.runner.get_page_source(), criteria)
            if not w_tgt:
                return None
            else:
                src_event = self.src_events[self.current_src_index]
                w_tgt['stepping_events'] = stepping
                w_tgt['package'] = self.runner.get_current_package()
                w_tgt['activity'] = self.runner.get_current_activity()
                w_tgt['event_type'] = src_event['event_type']
                w_tgt['score'] = WidgetUtil.weighted_sim(w_tgt, src_event)
                if src_event['action'][0] == 'wait_until_text_invisible':
                    # here, w_tgt is the nearest button to the text. Convert it to the oracle event
                    for k in w_tgt.keys():
                        if k not in ['stepping_events', 'package', 'activity', 'event_type', 'score']:
                            w_tgt[k] = ''

                if src_event['action'][0] == 'wait_until_text_presence':
                    # cache the closest button on the current screen for possible text_invisible oracle in the future
                    self.nearest_button_to_text = WidgetUtil.get_nearest_button(self.runner.get_page_source(), w_tgt)
                    self.nearest_button_to_text['activity'] = w_tgt['package']
                    self.nearest_button_to_text['package'] = w_tgt['activity']

                return w_tgt
        
        else:
            if self.steppings=={} or ppath0==[]:
                stepping=[]
            else:
                if ppath0 not in self.steppings.keys():
                    stepping=[]
                else:
                    stepping=self.steppings[ppath0][0]
                    print("stepping++",stepping)
            if stepping==[]:
                w_tgt=self.find_wtgt1(w_target,stepping)
                return w_tgt
            else:
                self.runner.perform_actions(self.tgt_events, reset=True, cgp=self.cgp)
                self.runner.perform_actions(stepping, require_wait=False, reset=False, cgp=self.cgp)
                self.cache_seen_widgets(self.runner.get_page_source(),self.runner.get_current_package(),self.runner.get_current_activity())
                w_tgt=self.find_wtgt1(w_target,stepping) 
                return w_tgt

    
    
    def find_wtgt1(self,w,stepping):

        attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password'})
        criteria = {k: w[k] for k in attrs_to_check if k in w}
        
        # for text_presence oracle, force the text to be the same as the src_event
        if self.src_events[self.current_src_index]['action'][0] == 'wait_until_text_presence':
            criteria['text'] = self.src_events[self.current_src_index]['action'][3]
        # for confirm email: if both prev and current src_action are input email
        if self.current_src_index > 0 and self.is_for_email_or_pwd(self.src_events[self.current_src_index-1],
                                                self.src_events[self.current_src_index]):
        # for the case of matching to the only one email field
            if StrUtil.is_contain_email(self.src_events[self.current_src_index]['action'][1]):
                criteria['text'] = self.runner.databank.get_temp_email(renew=False)
        w_tgt = WidgetUtil.locate_widget(self.runner.get_page_source(), criteria)
        # print("w_tgt",w_tgt)
        if w_tgt:
            src_event = self.src_events[self.current_src_index]
            w_tgt['stepping_events'] = stepping
            w_tgt['package'] = self.runner.get_current_package()
            w_tgt['activity'] = self.runner.get_current_activity()
            w_tgt['event_type'] = src_event['event_type']
            w_tgt['score'] = WidgetUtil.weighted_sim(w_tgt, src_event)
            if src_event['action'][0] == 'wait_until_text_invisible':
                # here, w_tgt is the nearest button to the text. Convert it to the oracle event
                for k in w_tgt.keys():
                    if k not in ['stepping_events', 'package', 'activity', 'event_type', 'score']:
                        w_tgt[k] = ''

            if src_event['action'][0] == 'wait_until_text_presence':
                # cache the closest button on the current screen for possible text_invisible oracle in the future
                self.nearest_button_to_text = WidgetUtil.get_nearest_button(self.runner.get_page_source(), w_tgt)
                self.nearest_button_to_text['activity'] = w_tgt['package']
                self.nearest_button_to_text['package'] = w_tgt['activity']

            return w_tgt
        

    def path_show(self,ppath):
    
        path_show = []
        for hop in ppath:
            if '(' in hop:
                if hop.startswith('D@'):
                    gui = ' '.join(hop.split()[:-1])
                else:
                    gui = self.rp.get_wName_from_oId(hop.split()[0])
                path_show.append(gui)
            else:
                path_show.append(StrUtil.get_activity((hop)))

        return path_show

    @staticmethod
    def fitness(events):
        for ei in events:
            if ei['event_type']=='stepping':
                continue
            if 'score' not in ei.keys():
                ei['score']=None
            if ei['score']==None:
                ei['score']=0

        gui_scores = [float(e['score']) for e in events if e['event_type'] == 'gui']
        oracle_scores = [float(e['score']) for e in events if e['event_type'] == 'oracle']
        gui = mean(gui_scores) if gui_scores else 0
        oracle = mean(oracle_scores) if oracle_scores else 0
        return 0.5*gui + 0.5*oracle

    def snapshot(self):
        with open(os.path.join(SNAPSHOT_FOLDER, self.config.id + '.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def check_mapped(self, match):
        tgt_idx = -1
        for i, e in enumerate(self.tgt_events):
            if e['class'] != 'android.widget.EditText' or 'send_keys' not in e['action'][0]:
                continue
            e_tgt_new_text = deepcopy(e)
            e_tgt_new_text['text'] = e_tgt_new_text['action'][1]
            # todo: ensure that e and match are on the same screen
            if WidgetUtil.is_equal(match, e) or WidgetUtil.is_equal(match, e_tgt_new_text):
                tgt_idx = i
                break
        if tgt_idx == -1:
            return False, -1, -1
        else:
            src_idx = -1
            for i_src, i_tgt in self.idx_src_to_tgt.items():
                if i_tgt == tgt_idx:
                    src_idx = i_src
                    break
            assert src_idx != -1
            return True, tgt_idx, src_idx

    def check_skipped(self, match):
        for skipped in self.skipped_match[self.current_src_index]:
            skipped_new_text = deepcopy(skipped)
            skipped_new_text['text'] = skipped_new_text['action'][1]
            if WidgetUtil.is_equal(match, skipped) or WidgetUtil.is_equal(match, skipped_new_text):
                return True
        return False

    def check_identical_src_widgets(self, src_idx1, src_idx2):
        """ True: treat two src widgets as the same, i.e., not to check identical mapping.
            e.g., a15-a11-b12 or a31-a32-b31 (for confirm email/password EditText)
        """
        if src_idx1 == -1 or src_idx2 == -1:
            return True
        src_e1 = self.src_events[src_idx1]
        src_e2 = self.src_events[src_idx2]
        src_classes_to_check = ['android.widget.EditText', 'android.widget.MultiAutoCompleteTextView']
        if src_e1['class'] in src_classes_to_check and src_e2['class'] in src_classes_to_check:
            if self.is_for_email_or_pwd(src_e1, src_e2):
                return True
            else:
                w1 = deepcopy(src_e1)
                w1['text'] = ''
                w2 = deepcopy(src_e2)
                w2['text'] = ''
                return WidgetUtil.is_equal(w1, w2)
        else:
            return True

    def is_for_email_or_pwd(self, src_e1, src_e2):
        if 'send_keys' in src_e1['action'][0] and 'send_keys' in src_e2['action'][0]:
            if src_e1['action'][1] == src_e2['action'][1]:
                if StrUtil.is_contain_email(src_e1['action'][1]) or \
                        src_e1['action'][1] == self.runner.databank.get_password():
                    return True
        return False

    def decay_by_distance(self, w_candidates, current_pkg, current_act):
        new_candidates = []
        for w, score in w_candidates:
            act_from = current_pkg + current_act
            act_to = w['package'] + w['activity']
            if act_from == act_to:
                d = 1
            else:
                potential_paths = self.cgp.get_paths_between_activities(act_from, act_to, self.consider_naf_only_widget)
                if not potential_paths:
                    d = 2
                else:
                    shortest_path, shortest_d = potential_paths[0], len(potential_paths[0])
                    for ppath in potential_paths[1:]:
                        if len(ppath) < shortest_d:
                            shortest_path, shortest_d = ppath, len(ppath)
                    d = len([hop for hop in shortest_path if '(' in hop or 'D@' in hop])  # number of GUI events
                    assert d >= 1
            new_score = score / (1 + math.log(d, 2))
            new_candidates.append((w, new_score))
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        if [s for w, s in w_candidates[:10]] != [s for w, s in new_candidates[:10]]:
            print('** Similarity rank changed after considering distance')
        return new_candidates


    def remove_repeat(self,results):
        
        resource_id_dict={}
        j=0
        while j<len(results):
            if results[j]['class']=='EMPTY_EVENT' or results[j]['class']=='SYS_EVENT':
                j=j+1
                continue
            id=results[j]['resource-id']
            if id not in resource_id_dict.keys():
                resource_id_dict.setdefault(id,[j])
            else:
                resource_id_dict[id].append(j)
            j=j+1
        print("results",results)
        print("resource_id_dict",resource_id_dict)


        # resource_id_dict {'com.rubenroy.minimaltodo:id/addToDoItemFAB': [0, 2, 5], 
        # 'com.rubenroy.minimaltodo:id/makeToDoFloatingActionButton': [1, 4, 6], 
        # 'com.rubenroy.minimaltodo:id/userToDoEditText': [3], 
        # 'com.rubenroy.minimaltodo:id/toDoListItemTextview': [7]}
        repeat=[]
        repeatlist1=[]
        for u in resource_id_dict.keys():
            if len(resource_id_dict[u])>1:
                repeat.append(u)
        print("repeat",repeat)
        resultsa=results
        if len(repeat)==1:
            if resource_id_dict[repeat[0]][0]==0:

                if len(resource_id_dict[repeat[0]])>2:
                    repeatlist1=resource_id_dict[repeat[0]][0:-1]
                if len(resource_id_dict[repeat[0]])==2:
                    repeatlist1=resource_id_dict[repeat[0]]

                if results[0]['class']=='EMPTY_EVENT':
                    a = 2
                    repeatlist2 = [1]
                    action0=results[1]['action']
                else:
                    a = 1
                    repeatlist2 = [0]
                    action0=results[0]['action']

                while a <= len(repeatlist1) - 1:
                    if repeatlist1[a] == repeatlist1[a - 1] + 1:
                        repeatlist2.append(a)
                        a = a + 1
                    else:
                        break
                repeatlist1 = repeatlist2

                for i in repeatlist1:
                    if results[i]['action']!=action0 or results[i]['event_type'] == "stepping":
                        repeatlist1=[]
                        break
                if len(repeatlist1)>0:
                    for i in repeatlist1:
                        # 将开头重复的gui事件置空
                        if results[i]['score']>0.25:
                            continue
                        resultsa[i]={"class": "EMPTY_EVENT","score": 0,"event_type": "gui"}
                        findnum = -1
                        stop = 0
                        # 同时更新对应的目标事件列表
                        for j in range(0,len(self.tgt_eventslist)):
                           findnum=findnum+len(self.tgt_eventslist[j])
                           if findnum>=i:
                                findnum = findnum - len(self.tgt_eventslist[j])
                                for m in range(0,len(self.tgt_eventslist[j])):
                                    findnum=findnum+1
                                    if findnum==i:
                                        self.tgt_eventslist[j][m] = {"class": "EMPTY_EVENT", "score": 0, "event_type": "gui"}
                                        stop = stop + 1
                                        break
                           if stop==1:
                                break


        if results[0]['class']!='EMPTY_EVENT' and results[1]['class']!='EMPTY_EVENT':
            if len(repeat)==2 and results[0]['resource-id']==repeat[0] and results[1]['resource-id']==repeat[1]:

                # repeat ['com.rubenroy.minimaltodo:id/addToDoItemFAB', 'com.rubenroy.minimaltodo:id/makeToDoFloatingActionButton']
                repeat1=repeat[::-1]
                # repeat1 ['com.rubenroy.minimaltodo:id/makeToDoFloatingActionButton','com.rubenroy.minimaltodo:id/addToDoItemFAB']


                for i in repeat1:
                    c=resource_id_dict[i]
                    repeatlist1.append(c[1:])
                # repeatlist1 [[4, 6], [2, 5]]
                m=0
                la=[]
                while m<len(repeatlist1)-1:
                    for k in repeatlist1[m]:
                        if k+1 in repeatlist1[m+1]:
                            la+=[k,k+1]
                    m=m+1
                # 对索引进行反转，使其从后往前删除
                for i in la[::-1]:
                    resultsa.pop(i)
                    findnum = -1
                    stop = 0
                    for k in range(0, len(self.tgt_eventslist)):
                        findnum = findnum + len(self.tgt_eventslist[k])
                        if findnum >= i:
                            print("findnum,i", findnum, i)
                            findnum = findnum - len(self.tgt_eventslist[k])
                            for m in range(0, len(self.tgt_eventslist[k])):

                                findnum = findnum + 1
                                print("findnum,i", findnum, i)
                                if findnum == i:
                                    self.tgt_eventslist[k].pop(m)
                                    stop = stop + 1
                                    break

                        if stop == 1:
                            break
                j=len(la)-1
                while j>=0:
                    resultsa.pop(j)
                    findnum = -1
                    stop = 0
                    for k in range(0, len(self.tgt_eventslist)):
                        findnum = findnum + len(self.tgt_eventslist[k])
                        if findnum >= j:
                            print("findnum,i", findnum, j)
                            findnum = findnum - len(self.tgt_eventslist[k])
                            for m in range(0, len(self.tgt_eventslist[k])):

                                findnum = findnum + 1
                                print("findnum,i", findnum, j)
                                if findnum == j:
                                    self.tgt_eventslist[k].pop(m)
                                    stop = stop + 1
                                    break

                        if stop == 1:
                            break
                    j=j-1
                resultsa.insert(0,{"class": "EMPTY_EVENT","score": 0,"event_type": "gui"})
                self.tgt_eventslist[0] = [{"class": "EMPTY_EVENT", "score": 0, "event_type": "gui"}]
                m = len(self.tgt_eventslist) - 1
                while m >= 0:
                    if self.tgt_eventslist[m] == []:
                        self.tgt_eventslist.pop(m)
                    m = m - 1
        return resultsa

    def mute(self):
        mutant = {'long_press': 'swipe_right', 'swipe_right': 'long_press'}
            
        self.src_events=self.mutate_src_action(mutant)
        print("self.src_events",self.src_events)
            
        self.f_prev_target = self.f_target
        self.prev_tgt_events = self.tgt_events
        self.tgt_events = []
        self.tgt_eventslist = []
        self.current_src_index = 0
        self.invalid_events = defaultdict(list)
        self.skipped_match = defaultdict(list)
        self.idx_src_to_tgt = {}
        is_explored = False
        self.tgt_events=self.gen_tarevents(is_explored)
        self.f_target = self.fitness(self.tgt_events)
        print(f'Current target events with fitness {self.f_target}:')
        return self.tgt_events

if __name__ == '__main__':
    # python Explorer.py a25-a22-b21 1 5723 emulator-5556 2>&1 | tee log\1-step\a25-a22-b21.txt
    
    config_id = sys.argv[1]
        # lookahead_step = int(sys.argv[2])
    print("config_id",config_id)
    appium_port ='4723'
    udid = 'emulator-5554'

    LOAD_SNAPSHOT = False
    # LOAD_SNAPSHOT = False
    if os.path.exists(os.path.join(SNAPSHOT_FOLDER, config_id + '.pkl')) and LOAD_SNAPSHOT:
        with open(os.path.join(SNAPSHOT_FOLDER, config_id + '.pkl'), 'rb') as f:
            explorer = pickle.load(f)
            explorer.runner = Runner(explorer.config.pkg_to, explorer.config.act_to, explorer.config.no_reset, appium_port, udid)
            # explorer.f_target = 0.55

    else:
        explorer = Explorer(config_id, appium_port, udid)

    t_start = time.time()
    # explorer.mutate_src_action({'long_press': 'swipe_right', 'swipe_right': 'long_press'})
    is_done, failed_step = explorer.run()
    if is_done:
        munu=0
        if munu==0:
            for i in explorer.tgt_events:
                if i['class']=='EMPTY_EVENT':
                    continue
                if i['event_type']=='gui':
                    if i['action']==['long_press'] or i['action']==['swipe_right']:
                        munu=munu+1
                        break

        if munu==1 and explorer.tgt_events[-1]['class']=='EMPTY_EVENT':
            explorer.tgt_events=explorer.mute()
           
        print('Finished. Learned actions')
        if explorer.f_prev_target > explorer.f_target:
            results = explorer.prev_tgt_events
            if munu==1:
                mutant = {'long_press': 'swipe_right', 'swipe_right': 'long_press'}
                explorer.src_events=explorer.mutate_src_action(mutant)

        else:
            results = explorer.tgt_events

        results1=[]
        results0=deepcopy(results)
        print("results0",results0)

        if results1!=[]:
            try:
                explorer.runner.perform_actions(results1)
                resultsfinal=results1
            except Exception as excep:
                resultsfinal = results0
                print("resultsfinal2",resultsfinal)
                print(f'Error when validating learned actions\n{excep}')
        else:
            resultsfinal =results

        if explorer.f_target<0.2:

            if resultsfinal[-1]['class']=='EMPTY_EVENT'and resultsfinal[-1]['event_type']=='oracle':
                resultsfinal=explorer.adj(resultsfinal)
            if 'stepping_events' in list(explorer.src_events[1].keys()) and explorer.src_events[1]['stepping_events']!=[] and resultsfinal[0]['class']!='EMPTY_EVENT':
                resultsfinal = explorer.adj(resultsfinal)
        else:
            if resultsfinal[-2]['class']=='EMPTY_EVENT'and resultsfinal[-2]['event_type']=='gui':
                resultsfinal=explorer.adj(resultsfinal)
            if 'stepping_events' in list(explorer.src_events[1].keys()):
                if explorer.src_events[1]['stepping_events']!=[] and resultsfinal[0]['class']!='EMPTY_EVENT':
                    resultsfinal = explorer.adj(resultsfinal)

        if resultsfinal[-1]['class'] != 'EMPTY_EVENT':
            if resultsfinal[-1]['action'][0] == 'wait_until_text_presence' and resultsfinal[-1]['action'][-1]!=explorer.src_events[-1]['action'][-1] or \
                    resultsfinal[-1]['action'][0] == 'wait_until_element_presence' and resultsfinal[-1]['action'][-1]!=explorer.src_events[-1]['action'][-1]:
                resultsfinal = explorer.adj(resultsfinal)
        trans_time=time.time() - t_start
        print(f'Transfer time in sec: {trans_time}')
        
    else:
        print(f'Failed transfer at source index {failed_step}')
        print(f'Transfer time in sec: {time.time() - t_start}')
        # resultsfinal = explorer.tgt_events
    Util.save_events(resultsfinal, config_id)
    f = open("b42.txt", "a")    # 打开outputfile.txt文件以便写入
    print(config_id,resultsfinal,trans_time, file=f)
    f.close   # 关闭文件

