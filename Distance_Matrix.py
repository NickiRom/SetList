# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import requests
import json
from json import load
from pprint import pprint 
import urllib2
from urllib2 import urlopen
import pandas as pd
from pandas import *
import numpy as np
import urllib
import csv
from collections import defaultdict
from Beats2EchoNest import beats2echonest

from scipy.spatial.distance import pdist, wminkowski, squareform
import matplotlib.pyplot as plt
import prettyplotlib as ppl

def Distance_Matrix(summarydf):
    
    #convert to list of rows (list of lists)
    for index in summarydf.ix[1]:
        if row[1] == " ":
            trow = [" "," "," "," "," "," "," "]
            print "crap"
        else:
            print row[2]
            trow = []
            trow.append(row[2])         #track_id
            #trow.append(float(row[5]))  #tempo
            #trow.append(float(row[6]))  #energy
            #trow.append(float(row[11])) #danceability
            #trow.append(float(row[14])) #loudness
            #trow.append(float(row[15])) #valence
            #list_of_songdata.append(trow)


    ranges = zip(*list_of_songdata)[1:]
    minimum = map(min, ranges)
    maximum = map(max, ranges)
    rangemap = [m-n for m,n in zip( maximum, minimum)]
    
    weights = [float(1/r) for r in rangemap]
    
    X = np.array(list_of_songdata)
    
    X1 = X[:, 1:]
    print X1
    
    distances = pdist(X1, wminkowski, 2, weights)
    distance_matrix= squareform(distances)
    print dist_matrix
    
    distancelist = []
    for index in range(0,17):
        distancelist.append(dist_matrix[index])
    
    return dist_matrix

'''with open("/home/vanessa/NHRInsightFL/Playlist1DistanceMatrixsmall.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(distance_matrix)'''



    

# <codecell>


# <codecell>

fig, ax = plt.subplots(1)
data = np.loadtxt(open("/home/vanessa/NHRInsightFL/Playlist1DistanceMatrixsmall.csv","rb"),delimiter=",",skiprows=1)
transformed=[]

print 
for line in data:
    transline = []
    for index, bit in enumerate(line):
        bity = -0.8*float(bit)+1
        #print bity
        transline.append(bity)
    transformed.append(transline)
transformed = np.array(transformed)

fig, ax = ppl.subplots(1)


ppl.pcolormesh(fig, ax, transformed)
fig.savefig('pcolormesh_prettyplotlib_default.png')

# <codecell>


