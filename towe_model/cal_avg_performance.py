import os
import numpy as np
home='./logs/'
files=os.listdir(home)
files.sort()
def cal_f(r):
    avg_p=[]
    avg_r=[]
    avg_f=[]
    num=0
    for f in files:
        if(r not in f):continue
        with open(os.path.join(home,f),'r') as f:
            lines=f.readlines()
            for i in range(len(lines)):
                line=lines[i]
                line=line.replace('p','').replace('r','').replace('f','').replace('\n','').replace(',','').replace(':','')
                line=line.split(' ')
                if(line[0]=='Best'):
                    cal_line=lines[i+3]
                    cal_line=cal_line.replace('p','').replace('r','').replace('f','').replace('\n','').replace(',','').replace(':','')
                    cal_line=cal_line.split(' ')
                    avg_p.append(float(cal_line[1]))
                    avg_r.append(float(cal_line[2]))
                    avg_f.append(float(cal_line[3]))
                    num+=1
        if(num==5):
            #print(avg_f)
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
