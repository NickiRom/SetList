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
import matplotlib as mpl
from matplotlib import pyplot as plt
import prettyplotlib as ppl
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from collections import defaultdict
from operator import itemgetter
import random
import collections, itertools
import string
import os



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
    
    access_token = '?access_token=y9sa6skwrxjcuwyagbu7ux7j'
    client_id = '&client_id=cu4dweftqe5nt2wcpukcvgqu'
    
    url = 'https://partner.api.beatsmusic.com/v1/api/playlists/' + beats_playlist + access_token
    response = urllib2.urlopen(url)
    json_obj = json.load(response)
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
        
        response = urllib2.urlopen(beats_url)
        json_obj = json.load(response)
     
            
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

            tempdf['artist']= json_obj['response']['songs'][0]['artist_name']
            tempdf['track_id']= json_obj['response']['songs'][0]['id']
            tempdf['song']=json_obj['response']['songs'][0]['title']
            playlist.append(json_obj['response']['songs'][0]['title'])
    
        df = df.append(tempdf, ignore_index = True)
        

    summarydf = pd.DataFrame(df, columns = columns)
    
    #get an original ordering of songs and artists to display next to transition matrix
    orig_artists_and_songs=[]
    artists = summarydf[:]['artist'].tolist()  #get list of artists
    songs = summarydf[:]['song'].tolist()   #get list of songs (in original order)
    length = len(songs)
    labels = string.uppercase[:length]
    orig_artists_and_songs = zip(labels, artists,songs)  #zip them together
    
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

    ppl.pcolormesh(fig, ax, transformed, xticklabels=labels, yticklabels=labels)
    ax.legend_ =None
    fig.savefig('app/static/'+str(filename), transparent=True)
    
    return songdatalist, dist_matrix, playlist, summarydf, orig_artists_and_songs

#------------------------------------------------------------------------------

def window(it, winsize, step=1):
    """Sliding window iterator."""
    it=iter(it)  # Ensure we have an iterator
    l=collections.deque(itertools.islice(it, winsize))
    while 1:  # Continue till StopIteration gets raised.
        yield tuple(l)
        for i in range(step):
            l.append(it.next())
            l.popleft()
#------------------------------------------------------------------------------

def DiGraph(songdatalist, dist_matrix, playlist, summarydf, filename):
    

    #convert to dataframe with trackIDs as columns
    columns = ['a','b','c','d','e','f','g','h','i','k','j','l','m','n','o','p','q','r']
    df = pd.DataFrame(dist_matrix, index = playlist, columns = playlist)

    
    index = 0
    row = 0
    
    tups = []
    cols = columns
    
    #put distance matrix into list of lists [[track1, track2, weight],...] for depth first search
    for index1, rows in enumerate(df):
        for index, cols in enumerate(df):
            mytups = [df.index[index1], df.columns[index], df.ix[index1][index]]
            tups.append(mytups)
    
    
    #save distance matrix as tuples before removing some values (next step)
    orig_tups = []    
    for tup in tups:
        orig_tup = (tup[0],tup[1],tup[2])
        orig_tups.append(orig_tup)
   
    
    #rank transition scores
    scores = []
    
    for item in tups:
        scores.append(item[2])
    
    scores = sorted(scores, reverse=True)
    
    #Remove all tups that represent a song transitioning to itself
    for worst in scores:
        for tup in tups:
            if tup[2] == 0:
                tups.remove(tup)
    

    #in preparation for removing worse than average values, find out distribution of scores
    scores = []
    
    for item in tups:
        scores.append(item[2])
    
    scores = sorted(scores, reverse=True)
    average_score = sum(scores)/float(len(scores)) 
    
    
    #find weight of the shuffled playlist
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
    
   
    #tups_weights = []  #get weights of all kept tups for shuffle validation

    #tups_weights.append(tup[2])
    
    DG=nx.DiGraph()
    DG.add_weighted_edges_from(tups)
    #start DFS from each node in DG. Order is the order of songs in a playlist starting from a node; Orderlist 
    orderlist=[]
    avg_edges=[]
    weightlistlist=[]
    
    
    for index, nodes in enumerate(DG): #start a DFS at each node

        order = list(nx.dfs_postorder_nodes(DG, nodes)) #order of the search is recorded as a list of nodes
        orderlist.append(order)  #added to a list of paths, each starting at a node

        weightlist = []
        songpairlist = list(window(order,2))

        for songpair in songpairlist:
            for tup in tups:
                if tup[0]==songpair[0]:
                    if tup[1]==songpair[1]:
                        weightlist.append(tup[2])  #add the path weight to the list of path weights
        weightlistlist.append(weightlist)
        avg_edge = sum(weightlist)/len(weightlist)  #edge weight per track in this playlist
        avg_edges.append(avg_edge)    #edge weight per track for all playlists; will be used to find the best playlist from all the n dfs trees
        
    min_edgepath = min(avg_edges)   #identifies the best playlist from average edgeweights for each dfs
    for i,j in enumerate(avg_edges):
        if j == min_edgepath:
            idxmin = i
            bestpath = orderlist[idxmin]  #finds which dfs is connected to the lowest average edgeweight and identifies that bestpath



    playlist_heat =  weightlistlist[idxmin]
    spacer = [None]*len(playlist_heat)

    comparison = zip(playlist_heat, spacer,spacer,spacer, shuffle)
    transformed = np.array(comparison)

    fig, ax = ppl.subplots(1)
    p = ax.pcolormesh(transformed, facecolor="black",vmin=0, vmax=1.6, cmap=mpl.cm.Reds)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.legend_ =None
    fig.savefig('app/static/comparison_'+str(filename), transparent=True)


    
    
    #add artist names to the bestpath list of songs
    df_row_list = list(summarydf.itertuples())  #change summary dataframe to a list of lists
    song_and_artist=[]
    songs_and_artists=[]
    for bestsongs in bestpath:    #for each song in the best path
        for songs in df_row_list: #match the song name to a row in the summary
            if songs[3]==bestsongs:
                song_and_artist = [songs[1], bestsongs]  #create a song and artist pair
        songs_and_artists.append(song_and_artist)
    
                
    avg_shuffle= sum(shuffleweight)/len(shuffleweight)  #finds the average weight of a shuffled playlist


    
    improvement = int((avg_shuffle - min_edgepath)/avg_shuffle*100)  #finds improvement in playlist transitions
    min_edgepath = round(min_edgepath, 2)
    avg_shuffle = round(avg_shuffle, 2)
    
    return min_edgepath, shuffle, avg_shuffle, improvement, songs_and_artists

def get_cached(query): 
    cached_file = query.encode('utf-8')+'.json'
    os.getcwd()
    print cached_file
    written = json.load(open('app/static/'+cached_file))
    avg_shuffle = written[0]['avg_shuffle']
    improvement = written[0]['improvement']
    min_edgepath = written[0]['min_edgepath']
    orig_artists_and_songs=written[0]['orig_artists_and_songs']
    playlist_id = written[0]['playlist_id']
    shuffle = written[0]['shuffled']
    songs_and_artists = written[0]['songs_and_artists']

    return songs_and_artists, min_edgepath, shuffle, avg_shuffle, orig_artists_and_songs, improvement


