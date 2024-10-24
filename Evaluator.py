import os
from csv import DictReader
# local import
from Util import Util
from WidgetUtil import WidgetUtil
from const import LOG_FOLDER


class Evaluator:
    def __init__(self, sol_file):
        assert os.path.exists(sol_file), "Invalid config file path"
        self.solution = {}
        with open(sol_file) as f:
            reader = DictReader(f)
            self.solution = [r for r in reader]
        self.res = {'gui': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                    'oracle': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
        self.finished = 0

    def get_all_config_ids(self, sol_file):
        bid = sol_file.split('-')[-1].split('.')[0]
        config_ids = {}
        for row in self.solution:
            aid_from, aid_to = row['aid_from'], row['aid_to']
            if aid_from not in config_ids:
                config_ids[aid_from] = set()
            config_ids[aid_from].add(aid_to)
        res = []
        for k, v_set in config_ids.items():
            for v_ele in v_set:
                res.append('-'.join([k, v_ele, bid]))
        return res

    def evaluate(self, config_id):
        events_from = Util.load_events(config_id, 'base_from')
        events_to = Util.load_events(config_id, 'base_to')

        events_gen = Util.load_events(config_id, 'generated')
        aid_from = config_id.split('-')[0]
        aid_to = config_id.split('-')[1]
        ans = {}
        for row in self.solution:
            if row['aid_from'] == aid_from and row['aid_to'] == aid_to:
                ans[int(row['step_from'])] = int(row['step_to'])
        idx_gen = 0
        events_pred = []
        gui_all_tps = 0
        gui_all_fps = 0
        gui_all_fns = 0
        gui_all_tns = 0

        oracle_all_tps = 0
        oracle_all_fps = 0
        oracle_all_fns = 0
        oracle_all_tns = 0


        for idx_from, src_event in enumerate(events_from):
            if idx_gen == len(events_gen):
                break
            while events_gen[idx_gen]['event_type'] == 'stepping':
                events_pred.append(events_gen[idx_gen])
                idx_gen += 1
            events_pred.append(events_gen[idx_gen])
            # print("ans[idx_from]",ans[idx_from])
            event_ans = events_to[ans[idx_from]] if ans[idx_from] > -1 \
                else {'class': 'EMPTY_EVENT', 'event_type': src_event['event_type']}
            gui_tps, gui_fps, gui_fns, gui_tns, oracle_tps, oracle_fps, oracle_fns, oracle_tns = self.judge(events_pred, event_ans, src_event['event_type'])
            
            gui_all_tps += gui_tps
            gui_all_fps += gui_fps
            gui_all_fns += gui_fns
            gui_all_tns += gui_tns

            oracle_all_tps += oracle_tps
            oracle_all_fps += oracle_fps
            oracle_all_fns += oracle_fns
            oracle_all_tns += oracle_tns

            events_pred = []
            idx_gen += 1
        if WidgetUtil.is_equal(events_gen[-1], events_to[-1], ignore_activity=True):

            self.finished += 1
        # print(self.res)
        # input()
        return gui_all_tps, gui_all_fps, gui_all_fns, gui_all_tns, oracle_all_tps, oracle_all_fps, oracle_all_fns, oracle_all_tns


    def judge(self, es_pred, e_ans, event_type):
        # calibrate unimportant text for judge
        gui_tps = 0
        gui_fps = 0
        gui_fns = 0
        gui_tns = 0

        oracle_tps = 0
        oracle_fps = 0
        oracle_fns = 0
        oracle_tns = 0

        for e in es_pred:
            if 'resource-id' in e and e['resource-id'].endswith('folder_name'):
                if e['text'] in ['Inbox ' + str(i) for i in range(1, 21)]:
                    e['text'] = 'Inbox'

        if event_type not in ['gui', 'oracle']:
            return 0, 0, 0, 0, 0, 0, 0, 0
        ignore_activity = False
        if 'action' in e_ans:
            if e_ans['action'][0] == 'wait_until_text_presence':
                ignore_activity = True
            if e_ans['content-desc'] == 'Navigate up':
                if e_ans['action'][0] == 'wait_until_element_presence' or e_ans['action'][0] == 'click':
                    ignore_activity = True
        cat = None
        if e_ans['class'] == 'EMPTY_EVENT':
            if all([e['class'] == 'EMPTY_EVENT' for e in es_pred]):
                cat = 'tn'
                if event_type == 'gui':
                    gui_tns += 1
                elif event_type == 'oracle':
                    oracle_tns += 1
                    # print("tn")
                    # print(es_pred)
                    # print(e_ans)
                # tns += 1
                # print("tn")
                # print(es_pred)
                # print(e_ans)

            else:
                cat = 'fp'
                # fps += 1
                if event_type == 'gui':
                    gui_fps += 1
                elif event_type == 'oracle':
                    oracle_fps += 1
                    # print("--fpe_ans", e_ans)
                    # print("--fpes_pred", es_pred)

        elif e_ans['class'] != 'EMPTY_EVENT':
            if all([e['class'] == 'EMPTY_EVENT' for e in es_pred]):
                cat = 'fn'
                if event_type == 'gui':
                    gui_fns += 1
                elif event_type == 'oracle':
                    oracle_fns+=1
                # fns += 1
            else:
                # print("e,e_ans",e,e_ans)
                if any([WidgetUtil.is_equal(e, e_ans, ignore_activity) for e in es_pred]):
                    # print("e,tp",e)
                    cat = 'tp'
                    # tps += 1
                    if event_type == 'gui':
                        gui_tps += 1
                    elif event_type == 'oracle':
                        oracle_tps += 1
                        # print("tppred",es_pred)
                        # print("tppred",e_ans)
                else:
                    if event_type == 'gui':
                        gui_fps += 1
                    elif event_type == 'oracle':
                        oracle_fps += 1
                        # print("gui-e_ans",e_ans)
                        # print("gui-es_pred",es_pred)
                        # print('--fp')
                    # if event_type == 'oracle':
                    #     print("oracle",e_ans)
                    #     print("oracle",es_pred)
                    cat = 'fp'
                    # fps += 1
        assert cat
        # self.res[event_type][cat] += 1
        return gui_tps, gui_fps, gui_fns, gui_tns, oracle_tps, oracle_fps, oracle_fns, oracle_tns
        # return self.res[event_type]['tp'] / (self.res[event_type]['tp'] + self.res[event_type]['fp']), \
        #         self.res[event_type]['tp'] / (self.res[event_type]['tp'] + self.res[event_type]['fn'])

    def output_res(self):
        label = ['tp', 'tn', 'fp', 'fn']
        print("self.res",self.res)
        for k, v in self.res.items():
            # print(k)
            res = [v[lbl] for lbl in label]
            # print(res)
            # print([n/sum(res) for n in res])
            # print(f'Precision: {res[0] / (res[0] + res[2])}. '
            #       f'Recall: {res[0] / (res[0] + res[3])}')


if __name__ == '__main__':
    # total = {
    #     'gui': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    #     'oracle': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    # }

    apppss = {'1':['a11','a12','a13','a14','a15'],'2':['a21','a22','a23','a25'],'3':['a31','a35'],'4':['a41','a43','a44'],'5':['a51','a52','a53','a54','a55']}


    solutions = [
        'solution/a1-b11.csv',
        'solution/a1-b12.csv',
        'solution/a4-b41.csv',
        'solution/a4-b42.csv',
        'solution/a5-b51.csv',
        'solution/a5-b52.csv',
        'solution/a3-b31.csv',
        'solution/a3-b32.csv',
        'solution/a2-b21.csv',
        'solution/a2-b22.csv',
    ]


    import pandas as pd
    dfr = pd.DataFrame()

    for sol in solutions:
        evaluator = Evaluator(sol)
        cids = evaluator.get_all_config_ids(sol)
        for cid in cids:
            # print(cid)
            source, target, test = cid.split('-')
            cat = source[1]
            if source not in apppss[cat] or target not in apppss[cat]:
                continue
            # input()
            gui_tps, gui_fps, gui_fns, gui_tns, oracle_tps, oracle_fps, oracle_fns, oracle_tns = evaluator.evaluate(cid)
            try:
                oracle_precision = oracle_tps/(oracle_tps+oracle_fps)
            except:
                oracle_precision = 'divisible by zero'
            try:
                oracle_recall = oracle_tps/(oracle_tps+oracle_fns)
            except:
                oracle_recall = 'divisible by zero'
            try:
                gui_precision = gui_tps/(gui_tps+gui_fps)
            except:
                gui_precision = 'divisible by zero'
            try:
                gui_recall = gui_tps/(gui_tps+gui_fns)
            except:
                gui_recall = 'divisible by zero'
            dic = {
                # 'source':source,'target':target,'test':test,
            'id':'-'.join(['c'+source[1],'t'+test[2],source,target]),
            'oracle_precision':oracle_precision, 'oracle_recall':oracle_recall,
            'gui_precision':gui_precision, 'gui_recall':gui_recall,
            'gui_tps':gui_tps,'gui_fps':gui_fps,'gui_fns':gui_fns,'gui_tns':gui_tns,
            'oracle_tps':oracle_tps,'oracle_fps':oracle_fps,'oracle_fns':oracle_fns,'oracle_tns':oracle_tns,'Success':'','Reduction':''}

            dfr = pd.concat([dfr,pd.DataFrame([dic])], ignore_index=True)
            
    print(dfr)
    dfr.to_csv('result.csv', index=False)
        # evaluator.output_res()
        # input()
        # print(f'Finished: {evaluator.finished}/{len(cids)}')
        # for event_type in ['gui', 'oracle']:
        #     for res in ['tp', 'tn', 'fp', 'fn']:
        #         total[event_type][res] += evaluator.res[event_type][res]
    
    # print(total)

    # print('\n*** Total *** ')
    # print(total)
    # for event_type in ['gui', 'oracle']:
    #     print(event_type.upper())
    #     print('Precision:', total[event_type]["tp"] / (total[event_type]["tp"] + total[event_type]["fp"]))
    #     print('Recall:', total[event_type]["tp"] / (total[event_type]["tp"] + total[event_type]["fn"]))
    #     print('Accuracy:', (total[event_type]["tp"] + total[event_type]["tn"]) / (total[event_type]["tp"] + total[event_type]["tn"]+total[event_type]["fp"] + total[event_type]["fn"]))
    # tp = total['gui']["tp"] + total['oracle']["tp"]
    # fp = total['gui']["fp"] + total['oracle']["fp"]
    # tn = total['gui']["tn"] + total['oracle']["tn"]
    # fn = total['gui']["fn"] + total['oracle']["fn"]
    # all = tp+fp+tn+fn
    # print(tp/all, tn/all, fp/all, fn/all)
    # print(tp/(tp+fp), tp/(tp+fn))

