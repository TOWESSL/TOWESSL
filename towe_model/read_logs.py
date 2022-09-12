import os 
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='senti')
args = parser.parse_args()

ori_result = {
    ds: {
        thr: {
            "p": [],
            "r": [],
            "f": [],
            "l": [],
        } for thr in ['0.9', '0.7', '0.5']
    } for ds in ['14res', '15res', '16res', '14lap'] 
}

senti_result = {
    ds: {
        thr: {
            senti_thr:{
                "p": [],
                "r": [],
                "f": [],
                "l": [],
            } for senti_thr in ['0.9', '0.7','0.5']
        } for thr in ['0.9', '0.7', '0.5']
    } for ds in ['14res', '15res', '16res', '14lap'] 
}

avg_result = {
    ds: {
        thr: {
            senti_thr:{
                "p": [],
                "r": [],
                "f": [],
                "l": [],
            } for senti_thr in ['0.9', '0.7', '0.5']
        } for thr in ['0.9','0.7','0.5']
    } for ds in ['14res', '15res', '16res', '14lap'] 
}


ulb_result = {
    ds: {
        ulb_size: {
            "p": [],
            "r": [],
            "f": [],
            "l": [],
        } for ulb_size in ['0','10000', '50000','100000', '150000', '200000']
    } for ds in ['14res', '15res', '16res', '14lap'] 
}


# for Senti && Ori
if args.mode == 'ori' or args.mode == 'senti' or args.mode == 'avg':
    for root, dirs, files in os.walk('./logs/', topdown=True):
        for name in files:
            file_name, _ = os.path.splitext(name)
            task = file_name.split('_')[0]
            if task == 'ori':
                _, ds, conf_thr, seed = file_name.split("_")
            elif task == 'senti':
                _, ds, conf_thr, senti_thr, seed = file_name.split("_")
            elif task == 'avg':
                 _, ds, conf_thr, senti_thr, seed = file_name.split("_")
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for i in range(len(lines)-1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith("Best"):
                        result = lines[i+3]
                        result = result.replace('TEST:', '').strip()
                        p, r, f, l = result.split(',')
                        p = p.strip().split(':')[-1]
                        r = r.strip().split(':')[-1]
                        f = f.strip().split(':')[-1]
                        l = l.strip().split(':')[-1]

                        if task == 'ori':
                            ori_result[ds][str(conf_thr)]['p'].append(p)
                            ori_result[ds][str(conf_thr)]['r'].append(r)
                            ori_result[ds][str(conf_thr)]['f'].append(f)
                            ori_result[ds][str(conf_thr)]['l'].append(l)
                        elif task == 'senti':
                            senti_result[ds][str(conf_thr)][str(senti_thr)]['p'].append(p)
                            senti_result[ds][str(conf_thr)][str(senti_thr)]['r'].append(r)
                            senti_result[ds][str(conf_thr)][str(senti_thr)]['f'].append(f)
                            senti_result[ds][str(conf_thr)][str(senti_thr)]['l'].append(l)
                        elif task == 'avg':
                            avg_result[ds][str(conf_thr)][str(senti_thr)]['p'].append(p)
                            avg_result[ds][str(conf_thr)][str(senti_thr)]['r'].append(r)
                            avg_result[ds][str(conf_thr)][str(senti_thr)]['f'].append(f)
                            avg_result[ds][str(conf_thr)][str(senti_thr)]['l'].append(l)

if args.mode == 'avg':
    for ds in avg_result:
        for conf_thr in avg_result[ds]:
            for senti_thr in avg_result[ds][conf_thr]:
                avg_result[ds][conf_thr][senti_thr]['avg'] = {}
                for key in avg_result[ds][conf_thr][senti_thr]:
                    if key != 'avg':
                        v = avg_result[ds][conf_thr][senti_thr][key] 
                        v = list(map(float, v))
                        if len(v) != 0:
                            avg = sum(v) / len(v)
                            avg_result[ds][conf_thr][senti_thr]['avg'][key] = avg
                        print('AVG: {ds}-{conf_thr}-{senti_thr}-{key}: {v}'.format(ds=ds, conf_thr=conf_thr, senti_thr=senti_thr, key=key, v=v))
                print('AVG: {ds}-{conf_thr}-{senti_thr}-AVG: {avg}'.format(ds=ds, conf_thr=conf_thr, senti_thr=senti_thr, avg=avg_result[ds][conf_thr][senti_thr]['avg']))

if args.mode == 'senti':
    for ds in senti_result:
        for conf_thr in senti_result[ds]:
            for senti_thr in senti_result[ds][conf_thr]:
                senti_result[ds][conf_thr][senti_thr]['avg'] = {}
                for key in senti_result[ds][conf_thr][senti_thr]:
                    if key != 'avg':
                        v = senti_result[ds][conf_thr][senti_thr][key] 
                        v = list(map(float, v))
                        if len(v) != 0:
                            avg = sum(v) / len(v)
                            senti_result[ds][conf_thr][senti_thr]['avg'][key] = avg
                        print('SENTI: {ds}-{conf_thr}-{senti_thr}-{key}: {v}'.format(ds=ds, conf_thr=conf_thr, senti_thr=senti_thr, key=key, v=v))
                print('SENTI: {ds}-{conf_thr}-{senti_thr}-AVG: {avg}'.format(ds=ds, conf_thr=conf_thr, senti_thr=senti_thr, avg=senti_result[ds][conf_thr][senti_thr]['avg']))

if args.mode == 'ori':
    for ds in ori_result:
        for conf_thr in ori_result[ds]:
                ori_result[ds][conf_thr]['avg'] = {}
                for key in ori_result[ds][conf_thr]:
                    if key != 'avg':
                        v = ori_result[ds][conf_thr][key] 
                        v = list(map(float, v))
                        if len(v) != 0:
                            avg = sum(v) / len(v)
                            ori_result[ds][conf_thr]['avg'][key] = avg
                        print('ORI: {ds}-{conf_thr}-{key}: {v}'.format(ds=ds, conf_thr=conf_thr,  key=key, v=v))
                print('ORI: {ds}-{conf_thr}-AVG: {avg}'.format(ds=ds, conf_thr=conf_thr, avg=ori_result[ds][conf_thr]['avg']))


# for ulb
if args.mode == 'ulb':
    for root, dirs, files in os.walk('./ulb_logs/', topdown=True):
        for name in files:
            file_name, _ = os.path.splitext(name)
            task = file_name.split('_')[0]
            _, ds, ulb_size, _, seed = file_name.split("_")

            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for i in range(len(lines)-1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith("Best"):
                        result = lines[i+3]
                        result = result.replace('TEST:', '').strip()
                        p, r, f, l = result.split(',')
                        p = p.strip().split(':')[-1]
                        r = r.strip().split(':')[-1]
                        f = f.strip().split(':')[-1]
                        l = l.strip().split(':')[-1]

                        ulb_result[ds][str(ulb_size)]['p'].append(p)
                        ulb_result[ds][str(ulb_size)]['r'].append(r)
                        ulb_result[ds][str(ulb_size)]['f'].append(f)
                        ulb_result[ds][str(ulb_size)]['l'].append(l)

    # Print ulb
    for ds in ulb_result:
        for ulb_size in ulb_result[ds]:
                ulb_result[ds][ulb_size]['avg'] = {}
                for key in ulb_result[ds][ulb_size]:
                    if key != 'avg':
                        v = ulb_result[ds][ulb_size][key] 
                        v = list(map(float, v))
                        if len(v) != 0:
                            avg = sum(v) / len(v)
                            ulb_result[ds][ulb_size]['avg'][key] = avg
                        print('ULB: {ds}-{ulb_size}-{key}: {v}'.format(ds=ds, ulb_size=ulb_size,  key=key, v=v))
                print('ORI: {ds}-{ulb_size}-AVG: {avg}'.format(ds=ds, ulb_size=ulb_size, avg=ulb_result[ds][ulb_size]['avg']))

