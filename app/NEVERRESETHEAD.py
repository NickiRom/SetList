# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


#given a playlist id from Beats, generate track IDs.  Output of list of track IDs will go to Beats2EchoNest.py.  setlist runs Beats2EchoNest, EN_id2summary, Distance Matrix, Thresholding

import requests
import json
from json import load
from pprint import pprint 
import urllib2
from urllib2 import urlopen
import pandas as pd
from pandas import *
import numpy as np
from numpy import *
import urllib
import scipy
from scipy import *
from scipy.spatial.distance import pdist, wminkowski, squareform
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from collections import defaultdict
from operator import itemgetter
import random

def setlist(beats_playlist):
    
    track_id = beatspl2tracks(beats_playlist)
    #print track_id
    
    EN_id_list = beats2echonest(track_id)
    #print EN_id_list
    
    songdatalist, dist_matrix, playlist= EN_id2summary(filename, EN_id_list)
    #print summarydf
    
    
    UTlist, orderlist = DiGraph(songdatalist, dist_matrix, playlist)
    
    
    
    
    
#--------------------------------------------------------------------
def beatspl2tracks(beats_playlist):
    
    access_token = '?access_token=cmgntmf8jasy56jntgf8myca'
    client_id = '&client_id=cu4dweftqe5nt2wcpukcvgqu'
    
    url = 'https://partner.api.beatsmusic.com/v1/api/playlists/' + beats_playlist + access_token
    response = requests.get(url)
    json_obj = json.loads(response.text)
    pprint(json_obj)
    datum = json_obj['data']['refs']['tracks']
    
    track_id = []
    
    for song in datum:
        t = song['id']
        track_id.append(t.encode('utf-8'))
    
    return track_id

#--------------------------------------------------------------------
def beats2echonest(track_id):
    beats_url = 'https://partner.api.beatsmusic.com/v1/api/tracks/'

    #initialize list of identifier dicts
    identifier = []
    EN_id_list = []
    
    for tracks in track_id:
        beats_url = 'https://partner.api.beatsmusic.com/v1/api/tracks/'
        #print tracks
        query = tracks + "?"
        client_id = 'client_id=cu4dweftqe5nt2wcpukcvgqu'
        beats_url = beats_url + query + client_id
        #print beats_url
        
        response = requests.get(beats_url)
        json_obj = json.loads(response.text)
     
            
        trackname = json_obj['data']['title'].encode('utf-8')
        artist = json_obj['data']['artist_display_name'].encode('utf-8')
        duration = json_obj['data']['duration']
        min_duration = int(duration)*0.95
        max_duration = int(duration)*1.05
        
        tidentifier = {'artist':artist,'title':trackname, 'max_duration':max_duration, 'min_duration':min_duration}
        tidentifier = urllib.urlencode(tidentifier)
        identifier.append(tidentifier)
    
    #search for track in Echonest
    EN_id = []
    
    for codes in identifier:
        #print tracks
        apikey = 'W89S7QJCCHFARWJGD'
        jsonformat = '&format=json&results=1&'
        summary_request = '&bucket=audio_summary'
        
        url ='http://developer.echonest.com/api/v4/song/search?api_key=' + apikey + jsonformat + codes+  summary_request        
        

        response = urllib2.urlopen(url)

        json_obj = json.load(response)
        if len(json_obj['response']['songs'])==0:
            continue
            #EN_id = ' '
        else:
            EN_id = json_obj['response']['songs'][0]['id']
        EN_id_list.append(EN_id.encode('utf-8'))
    
    
    return EN_id_list
#--------------------------------------------------------------------



def EN_id2summary(filename, EN_id_list):
    #set up dataframe for collection
    df = pd.DataFrame({'artist': [], 'track_id':[], 'song':[],'key':[], 'tempo':[], 'energy':[], 'liveness':[], 'analysis_url':[], 'speechiness':[], 'acousticness':[], 'danceability':[], 'time_signature':[], 'duration':[], 'loudness':[], 'valence':[], 'mode':[]})

    columns = ['artist','track_id','song','key', 'tempo', 'energy', 'liveness', 'analysis_url', 'speechiness', 'acousticness', 'danceability', 'time_signature', 'duration', 'loudness', 'valence', 'mode']
    playlist = []
    
    for song in EN_id_list:
        if song == " ":
            tempdf = pd.DataFrame([(" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," ")], index = [0], columns = columns)
        else:
            apikey = 'W89S7QJCCHFARWJGD'
            jsonformat = '&format=json&results=1&'
            summary_request = '&bucket=audio_summary'
            id_url = 'id=' + str(song)
            
            url ='http://developer.echonest.com/api/v4/song/profile?api_key=' + apikey + jsonformat + id_url+  summary_request        
            response = urllib2.urlopen(url)
            json_obj = json.load(response)
        
            EN_id = json_obj['response']['songs'][0]['id']
        
            tempdict = json_obj['response']['songs'][0]['audio_summary']
            tempdf = pd.DataFrame(tempdict, index = [1])

            tempdf['artist']= json_obj['response']['songs'][0]['artist_id']
            tempdf['track_id']= json_obj['response']['songs'][0]['id']
            tempdf['song']=json_obj['response']['songs'][0]['title']
            playlist.append(json_obj['response']['songs'][0]['title'])
    
        df = df.append(tempdf, ignore_index = True)
        

    summarydf = pd.DataFrame(df, columns = columns)

    songdatalist = []

    #convert to list of rows (list of lists)
    for i in summarydf.index:
        row = summarydf.ix[i]
        
        rowlist = []
        
        if row['tempo'] == " ":
            rowlist = [" "," "," "," "," "," ", " "]
        else:
            rowlist = [row ['song'], row['track_id'], row['tempo'],row['energy'],row['danceability'],row['loudness'],row['valence']]#track_id
            #print rowlist
            
        songdatalist.append(rowlist)


    ranges = zip(*songdatalist)[2:]
    #print ranges
    minimum = map(min, ranges)
    maximum = map(max, ranges)
    rangemap = [m-n for m,n in zip( maximum, minimum)]
    
    weights = [float(1/r) for r in rangemap]
    
    X = np.array(songdatalist)
    
    X1 = X[:, 2:]
    #print X1
    
    distances = pdist(X1, wminkowski, 2, weights)
    dist_matrix= squareform(distances)
    #print dist_matrix
    
    distancelist = []
    for index in range(0, len(dist_matrix)):
        distancelist.append(dist_matrix[index])

    transformed = np.array(distancelist)

    fig, ax = ppl.subplots(1)

    ppl.pcolormesh(fig, ax, transformed)
    ax.legend_ =None
    fig.savefig('app/static/'+str(filename))
    
    return songdatalist, dist_matrix, playlist


#------------------------------------------------------------------------------

def DiGraph(songdatalist, dist_matrix, playlist):
    

    #convert to dataframe with trackIDs as columns
    columns = ['a','b','c','d','e','f','g','h','i','k','j','l','m','n','o','p','q','r']
    df = pd.DataFrame(dist_matrix, index = playlist, columns = playlist)
    #print df
    
    index = 0
    row = 2
    
    tups = []
    cols = columns
    
    #put distance matrix into list of lists [[track1, track2, weight],...] for depth first search
    for index1, rows in enumerate(df):
        for index, cols in enumerate(df):
            mytups = [df.index[index1].encode('ascii', 'ignore'), df.columns[index].encode('ascii', 'ignore'), df.ix[index1][index]]
            tups.append(mytups)
    #transform weights to create a higher penalty for higher weights
    orig_tups = tups
    scores = []
    
    for item in tups:
        scores.append(item[2])
    
    scores = sorted(scores, reverse=True)
    
    average_score = sum(scores)/float(len(scores))    

    for worst in scores:
        for tup in tups:
            if tup[2] == 0:
                tups.remove(tup)

    shuffleweight=[]
    mintup = 5
    maxtup = 0
    
    for tup in tups:
        if tup[0] != tup[1]:
            shuffleweight.append(tup[2])  #add a weight to the shuffle collection
            if tup[2] < mintup:   #find the minimum weight in the whole graph
                mintup = tup[2]
            if tup[2] > maxtup:   #find the maximum weight in the whole graph
                maxtup = tup[2]

    shufflerange = maxtup - mintup

    shuffle = random.sample(shuffleweight, len(playlist))

    # get an idea of the distribution of transition scores
    
    '''show histogram of 
    plt.hist(p, bins = 20, cumulative=True)
    plt.show()
    '''
    
    #prune edges from graph by removing lists in the edges list
    for worst in scores:
        for tup in tups:
            if tup[2] >= average_score:
                tups.remove(tup)
                
    tups_weights = []  #get weights of all kept tups for shuffle validation
    for tup in tups:
        tups_weights.append(tup[2])
    
    DG=nx.DiGraph()
    DG.add_weighted_edges_from(tups)
    #start DFS from each node in DG. Order is the order of songs in a playlist starting from a node; Orderlist 
    orderlist=[]
    avg_edges=[]
    weightlistlist=[]
    
    
    for index, nodes in enumerate(DG): #start a DFS at each node
        order = list(nx.dfs_edges(DG, nodes)) #order of the search is recorded for this node
        orderlist.append(order)  #added to a list of paths
        weightlist = []
        for song in order:
            for tup in tups:
                if tup[0]==song[0]:
                    if tup[1]==song[1]:
                        weightlist.append(tup[2])  #get the weight of the edges
    
        weightlistlist.append(weightlist)
        avg_edge = sum(weightlist)/len(weightlist)  #edge weight per track in this playlist
        avg_edges.append(avg_edge)    #edge weight per track for all playlists
        
 
    #min(enumerate(a), key=itemgetter(1))[0]
    minval, idxmin = min((val, idx) for (idx,val) in enumerate(avg_edges))

    bestlist = orderlist[idxmin]
    bestpath = []
    for songs in bestlist:
        bestpath.append(songs[0])
    
                
    avg_shuffle= sum(shuffleweight)/len(shuffleweight)


    
    improvement = int((avg_shuffle - minval)/avg_shuffle*100)
    minval = round(minval, 2)
    avg_shuffle = round(avg_shuffle, 2)
    
    return bestpath, minval, shuffle, avg_shuffle, improvement, orig_tups




