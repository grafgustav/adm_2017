# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:54:08 2017

@author: Niek
"""
import numpy as np
import random
import pandas as pd
from scipy.spatial.distance import pdist, jaccard
from scipy.spatial.distance import squareform
from collections import defaultdict

a = np.repeat(list(range(20)), 5)
a = np.array(random.sample(list(a), len(a)))

n = 100
Dataset = (np.random.randint(1, 8, 2000)).reshape(n, 20)
B = 20
R = n/B
Test = pd.DataFrame(Dataset[np.where(a == 1)[0]])
v = Test.values              
#lt = pd.DataFrame((v[:, None] == v.T))
#np.fill_diagonal(lt.values, 0)
res = 1 - pdist(Test.T, 'jaccard')
squareform(res)
distance = np.triu(pd.DataFrame(squareform(res), index = Test.T.index, 
                        columns = Test.T.index), k = 0)
distance = np.round(distance, decimals = 1)



d = defaultdict(list)
for pos, val in np.ndenumerate(distance):
    if val:
        d[val].append((pos[0], pos[1]))
        

print(d)
