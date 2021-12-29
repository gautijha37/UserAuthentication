# import necessary libraries
import os
import string
from datetime import datetime
import numpy as np
import itertools

# This is a script to extract features of the Keylog data. v.i.z., Keyholdtime and Key pair Latency

def timediff(t1,t2):
    #t1=(t1 + '000')
    #t2=(t2 + '000')
    day1=datetime.strptime(t1, "%d:%m:%Y:%H:%M:%S:%f")
    day2=datetime.strptime(t2, "%d:%m:%Y:%H:%M:%S:%f")
    sec = (day2-day1).total_seconds()
    return(sec)

def extract_keylog_features(path):
    lines = [line.rstrip('\t\n') for line in open(path)]
    f_list = [event.split('\t') for event in lines]
    #lines2 = [line.rstrip('\t\n') for line in open('1.txt')]
    #f_list2 = [event.split('\t') for event in lines2]
    alp = list(string.ascii_lowercase)
    alp2 = list(map(''.join,itertools.combinations(string.ascii_lowercase,2)))

    KeyUps = [x for x in f_list if 'KeyUp' in x]
    KeyDowns = [x for x in f_list if 'KeyDown' in x]

    tups =  [item[2] for item in KeyUps]# if item[1] in alp]
    tdowns =  [item1[2] for item1 in KeyDowns]# if item1[1] in alp]
    try:
        letterup =  [item[1].upper() for item in KeyUps]# if item[1] in alp]
    except: 
        pass

    try:
        letterdown = [item1[1].upper() for item1 in KeyDowns]# if item1[1] in alp]
    except:
        pass

    features = []
    for i in range(0,len(tups)-1):
        t = i

        t1 = tdowns[i] #timestamp of downpress

        if letterup[t] != letterdown[i]: #Simultaneous press
            j = i

            if i == len(tups)-1:
                j = 0
            while j<len(tups)-1 and letterdown[i]!= letterup[j] and i!=len(tups)-1: #skip inbetween to go to the release key
                j = j+1

            tj = tups[j]
            k = i

            if i == 0:
                k = len(tups)-1
            while k>=1 and letterdown[i]!= letterup[k] and i!=0: # repeat same key
                k = k-1

            tk = tups[k]


            if timediff(t1,tk)>0 and timediff(t1,tj)>0 :
                if abs(j-i)<abs(i-k):
                    t = j
                else:
                    t = k

            elif timediff(t1,tk)<0 :
                t = j
            else:
                t = k

        t2 = tups[t]        # release time stamp



        if i!=len(tups)-1:

            t3 = tdowns[i+1]  # Next key time stamp
            latency = timediff(t1,t3)

            lat = letterdown[i]+letterdown[i+1],latency
            features.append(lat)

        hold_time = timediff(t1,t2)
        hold = letterdown[i],hold_time
        features.append(hold)
        
    return features