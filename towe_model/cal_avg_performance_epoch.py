import os
import numpy as np
home='./logs/'
files=os.listdir(home)
files.sort()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

def cal_f(r):
    avg_p=[]
    avg_r=[]
    avg_f=[]
    e_list=[]
    num=0
    for f in files:
        if(r not in f):continue
        best_p=0
        best_r=0
        best_f=0
        dev_f=0
        best_e=0
        with open(os.path.join(home,f),'r') as f:
            lines=f.readlines()
            for i in range(len(lines)):
                line=lines[i]
                line=line.replace('p','').replace('r','').replace('f','').replace('\n','').replace(',','').replace(':','')
                line=line.split(' ')
                if(line[0]=='Best'):continue
                if('Eoch' in line[0] and len(line)==1):
                    if(float(line[0][4:])>=args.epochs):continue
                    dev_line=lines[i+1]
                    dev_line=dev_line.replace('p','').replace('r','').replace('f','').replace('\n','').replace(',','').replace(':','')
                    dev_line=dev_line.split()
                    cal_line=lines[i+2]
                    cal_line=cal_line.replace('p','').replace('r','').replace('f','').replace('\n','').replace(',','').replace(':','')
                    cal_line=cal_line.split(' ')
                    if(dev_f<=float(dev_line[3])):
                        best_e=float(line[0][4:])
                        best_p=float(cal_line[1])
                        best_r=float(cal_line[2])
                        best_f=float(cal_line[3])
        e_list.append(best_e)
        avg_p.append(best_p)
        avg_r.append(best_r)
        avg_f.append(best_f)
        num=len(avg_f)
        if(num==5):
            print(e_list)
            print(avg_f)
            print(r,'| %.2f | %.2f | %.2f | %.2f | %.2f | %.2f ' %(np.mean(avg_p)*100,np.mean(avg_r)*100,np.mean(avg_f)*100,np.std(avg_p)*100,np.std(avg_r)*100,np.std(avg_f)*100))
def find_roots():
    res=[]
    for f in files:
        root=f[:len(f)-6]
        if(root not in res):
            res.append(root)
    return res
roots=find_roots()
for r in roots:
    cal_f(r)
