"""
Created on Tue Nov 23 11:55:20 2021
@author: diego
"""

import pandas as pd
import numpy as np
import requests as rq
import folium
import webbrowser
from geopy.geocoders import Nominatim
import time
import json
from json import loads, dumps
from ast import literal_eval
import os

##########################################################################################
#########################  DOCKER SERVER on PORT 5000  ###################################
##########################################################################################

# before to go please start the map service on docker with the following command
# docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --algorithm mld /data/italy-latest.osrm
# tipo di servizio: 'OPENROUTESERVICE' si collega a https://openrouteservice.org/
# 'LOCAL' si collega al server locale installato tramite osrm-backend project
route_service = 'LOCAL'
# Inserire qui la propria api key ottenuta da https://openrouteservice.org/
# serve solo se il tipo di servizio è 'OPENROUTESERVICE
myApyKey = '5b3ce3597851110001cf62488cc89bf95d4b4c019b17589365548095'
DEBUG = False


def getPath(long1, lat1, long2, lat2):
    if route_service == 'OPENROUTESERVICE':
        headers = {
            'Accept': 'application/json; charset=utf-8'
        }
        myurl = 'https://api.openrouteservice.org/directions?api_key=' + myApyKey + '&coordinates=' + str(
            long1) + ',' + str(lat1) + '%7C' + str(long2) + ',' + str(lat2) + '&profile=driving-car'
        response = rq.request(url=myurl, headers=headers, method='GET')
        return response
    elif route_service == 'LOCAL':
        myurl = 'http://127.0.0.1:5000/route/v1/driving/'
        myurl += str(long1) + ',' + str(lat1) + ';' + str(long2) + ',' + str(lat2)
        params = (
            ('steps', 'false'),
        )
        response = rq.get(myurl, params=params)
        return response
    else:
        return 'ERROR'


def getDistance(response):
    a = dict(response.json())
    if DEBUG:
        print(a)
    if route_service == 'OPENROUTESERVICE':
        return a['routes'][0]['summary']['duration']
    elif route_service == 'LOCAL':
        dist = a['routes'][0]['distance']
        return dist
    else:
        return 'ERROR'


def getTime(response):
    a = dict(response.json())
    if DEBUG:
        print(a)
    if route_service == 'OPENROUTESERVICE':
        return a['routes'][0]['summary']['duration']
    elif route_service == 'LOCAL':
        duration = a['routes'][0]['duration']
        return duration
    else:
        return 'ERROR'


def get_ODbyOSM(nodes):
    df = nodes
    n = len(df.index)

    A = df.values

    inst_data = []
    disdur = {}

    j = 0
    for i in range(n):
        print(str(i * j) + '....' + str(i) + '/' + str(n))
        id_from = A[i][4]
        id_from_name = A[i][0]
        ntype_from = A[i][3]
        for j in range(n):
            id_to = A[j][4]
            id_to_name = A[j][0]
            ntype_to = A[j][3]
            long2 = A[i][2]
            lat2 = A[i][1]
            long1 = A[j][2]
            lat1 = A[j][1]
            if i != j:
                response = getPath(long1, lat1, long2, lat2)
                dis = round(getDistance(response) / 1000, 2)
                dur = round((getTime(response) / 60) * (1.2), 2)
            else:
                dis = 0
                dur = 0

            disdur[(id_from, id_to)] = {'distance': dis, 'duration': dur}
            inst_data.append(
                [id_from_name, id_to_name, id_from, ntype_from, id_to, ntype_to, lat2, long2, lat1, long1, dis,
                 dur])

    inst_data = pd.DataFrame(inst_data,
                             columns=['id_from_name', 'id_to_name', 'id_from', 'ntype_from', 'id_to', 'ntype_to',
                                      'lat_from',
                                      'long_from', 'lat_to', 'long_to', 'distance [km]', 'duration [min]'])

    return inst_data, disdur


def get_folium_map(nodes):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.93989132187777
    Long_c = 12.57960310938201

    # creo la mappa centrata sul centro mappa
    mappa = folium.Map(location=[Lat_c, Long_c], zoom_start=4)

    colors = {'client': 'red', 'facility': 'blue', 'depot': 'black'}

    nodes_list = nodes.values.tolist()
    for node in nodes_list:
        node_name = node[0]
        lat = node[1]
        long = node[2]
        location = ([lat, long])
        node_type = node[3]
        node_color = colors[node_type]

        """ aggiungo i marker dei nodi"""
        folium.Marker(
            location=location,
            popup=node_name,
            icon=folium.Icon(color=node_color, icon='user', prefix='fa')
        ).add_to(mappa)

    output_file = "map.html"
    mappa.save(output_file)
    webbrowser.open(output_file, new=2)  # open in new tab

    return mappa._repr_html_()


def get_node_region(lat, long):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(str(lat) + "," + str(long))
    regione = location.raw['address']['state']

    return regione

def create_instance(clients, facilities, NF, NC, ND, random_state):
    facilities = facilities.sample(n=NF, random_state=random_state)
    clients = clients.sample(n=NC, random_state=random_state + 1)
    depots = clients.sample(n=ND, random_state=random_state - 1)
    depots['node_type'] = 'depot'

    nodes = facilities.append(clients)
    nodes = nodes.append(depots)

    nodes.reset_index(inplace=True)
    nodes['node'] = nodes.index
    nodes.drop(columns=['level_0', 'index'], inplace=True)

    instace_df, disdur = get_ODbyOSM(nodes)

    return nodes, disdur


def save_instance(disdur,instance_name):
    # save: convert each tuple key to a string before saving as json object
    dict_tosave = dumps({str(k): v for k, v in disdur.items()})
    # save
    with open('disdur_'+instance_name+'.json', 'w') as fp:
        json.dump(dict_tosave, fp)

    return

def write_json_instance(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise

    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

def load_json_instance(target_path, target_file):
    # load in two stages:
    # Opening JSON file
    with open(os.path.join(target_path, target_file), 'r') as json_file:
        data = json.load(json_file)

    data['disdur_dict'] = loads(data['disdur_dict'])

    # (ii) convert loaded keys from string back to tuple
    data['disdur_dict'] = {literal_eval(k): v for k, v in data['disdur_dict'].items()}

    return data

if __name__ == '__main__':

    clients = pd.read_excel('All_clients.xlsx', index_col=[0])
    facilities = pd.read_excel('All_facilities.xlsx', index_col=[0])

    # set instance name, number of clients, facilities and depot for new instance creation

    # NF number of facilities
    # NC number of clients
    # ND number of depots
    # NV number of vehicles

    # sets dimensions are (NF [number of facilities], NC [number of clients], ND [number of depots], NV [number of vehicles])
    sets_dimension_list = [(5,15,2,4), (10,30,5,8), (15,60,8,12)]

    instance_num = 1

    size_dimension = 0
    for sets_dimension in sets_dimension_list:
        NF = sets_dimension[0]
        NC = sets_dimension[1]
        ND = sets_dimension[2]
        NV = sets_dimension[3]
        for instance in range(size_dimension*instance_num + 1, size_dimension*instance_num + instance_num+1):
            instance_name = 'inst_#'+str(instance)
            random_state = instance
            nodes, disdur = create_instance(clients, facilities, NF, NC, ND, random_state)

            nodes_info = nodes.to_dict()
            inst_data = {'NF': NF, 'NC': NC, 'ND': ND, 'NV': NV}
            disdur_tosave = dumps({str(k): v for k, v in disdur.items()})
            instance_data = {'nodes_info': nodes_info, 'inst_data': inst_data, 'disdur_dict' : disdur_tosave}

            write_json_instance('./instances', instance_name+'.json', instance_data)
            load_json_instance('./instances',  instance_name+'.json')

        size_dimension += 1



    display_map = False
    if display_map:
        instance_name = 'inst_#'+str(1)
        instance_data = load_json_instance('./instances', instance_name + '.json')
        nodes = pd.DataFrame.from_dict(instance_data[nodes_info])  ## FIX !!
        nodes.lat = pd.to_numeric(nodes.lat)
        nodes.long = pd.to_numeric(nodes.long)
        start = time.time()
        print('start map creation')
        mappa = get_folium_map(nodes)
        print('map created in ' + str(time.time() - start))