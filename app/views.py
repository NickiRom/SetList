import os
from flask import Flask, render_template, request, session, url_for, jsonify
from app import app, host, port, user, passwd, db
from app.helpers.database import con_db
import pymysql
from NEVERRESETHEAD import beatspl2tracks, beats2echonest, EN_id2summary, DiGraph, get_cached
import csv
import scipy
from scipy import stats


# To create a database connection, add the following
# within your view functions:
# con = con_db(host, port, user, passwd, db)

# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def index():
    # Renders index.html.

    return render_template('index.html')

app.secret_key = 'junk'

@app.route('/getplaylistinfo', methods=['POST'])
def getplaylistinfo():
    
    #for cached playlists:
    
    request.form['query']   
    select_query = request.form['query'] 
 

    beatstracks = beatspl2tracks(query)
    entracks = beats2echonest(beatstracks)  
    
    filename = 'request_' + query
    songdatalist, dist_matrix, playlist = EN_id2summary(filename, entracks)
    bestpath, minval, shuffle, avg_shuffle, improvement, orig_tups= DiGraph(songdatalist, dist_matrix, playlist, summarydf, filename)
    
    return render_template('getplaylistinfo.html',  query=query, playlist=bestpath)

app.secret_key='junk'

@app.route('/generatedistance', methods=['POST'])
def generatedistance():

    request.form['query']   
    query = request.form['query'] 
    
    path = ''
    cached_plid= ['pl170577122547466496','pl147773287731036416','pl151894969496371456', 'pl151764388523540736','pl152858163299746048','pl196764123525021952','pl196411940837261312', 'pl197125380790813184']
    for plids in cached_plid:
        if query == plids:
            path = "cached"
    if path=="cached":
        songs_and_artists, min_edgepath, shuffle, avg_shuffle, orig_artists_and_songs, improvement = get_cached(query)
        filename = 'request_' + query + '.png'
        url = url_for('static', filename =  'request_' + query + '.png')
        url2 = url_for('static', filename="comparison_"+filename)
    else:
        beatstracks = beatspl2tracks(query)
        entracks = beats2echonest(beatstracks)
        filename = 'request_' + query + '.png'
        songdatalist, dist_matrix, playlist, summarydf, orig_artists_and_songs = EN_id2summary(filename, entracks)
        min_edgepath, shuffle, avg_shuffle, improvement, songs_and_artists = DiGraph(songdatalist, dist_matrix, playlist, summarydf, filename)
        url = url_for('static', filename=filename)
        url2 = url_for('static', filename="comparison_"+filename)

    artists = []
    for song in songs_and_artists:
        artists.append(song[1])
    beats_img = "https://api.beatsmusic.com/api/playlists/"+query+"/images/default?size=large"

    
    mean_shuff, var_shuff, std_shuff = scipy.stats.bayes_mvs(shuffle, alpha=0.95)
    
    
    return render_template('generatedistance.html', query=query, beats_img=beats_img, url = url, url2=url2, minval=min_edgepath, shuffle=shuffle, avg_shuffle=avg_shuffle, improvement=improvement, songs_and_artists=songs_and_artists, orig_artists_and_songs=orig_artists_and_songs, mean_shuff=mean_shuff)
    
@app.route('/progressbar')
def progressbar():
    request.form['query']   
    query = request.form['query'] 

    beatstracks = beatspl2tracks(query)
    entracks = beats2echonest(beatstracks)

    return render_template('progress.html', query=query, beatstracks = beatstracks, entracks = entracks)

@app.route('/home')
def home():
    # Renders home.html.
    return render_template('home.html')

@app.route('/slides')
def about():
    # Renders slides.html.
    return render_template('slides.html')

@app.route('/author')
def contact():
    # Renders author.html.
    return render_template('author.html')

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
