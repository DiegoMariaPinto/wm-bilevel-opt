import pandas as pd
import folium
import webbrowser
import time
import json
from Get_instances import load_json_instance
def get_folium_map(nodes):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.94298561949368
    Long_c = 12.60683386876551

    tiles = None
    # creo la mappa centrata sul centro mappa
    m = folium.Map(location=[Lat_c, Long_c], tiles = tiles, zoom_start=10)


    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
    # tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png"

    tile_layer = folium.TileLayer(
        tiles= tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    geojson_filename = 'limits_IT_regions.geojson'
    with open(geojson_filename, 'r') as geojson_file:
        region_borders_layer = json.load(geojson_file)

    geojson_filename = 'limits_IT_provinces.geojson'
    with open(geojson_filename, 'r') as geojson_file:
        provinces_borders_layer = json.load(geojson_file)

    style =  {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(region_borders_layer, style_function=lambda x: style).add_to(m)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(provinces_borders_layer, style_function=lambda x: style).add_to(m)

    folium.LayerControl().add_to(m)

    colors = {'client': 'green',    'facility': 'orange', 'depot': 'blue'}
    icons  = {'client': 'refresh',  'facility': 'recycle','depot': 'truck'}

    nodes_list = nodes.values.tolist()
    for node in nodes_list:
        node_name = node[0]
        lat  = node[1]
        long = node[2]
        location   = ([lat, long])
        node_type  = node[3]
        node_color = colors[node_type]
        node_icon  = icons[node_type]

        """ aggiungo i marker dei nodi"""
        folium.Marker(
            location=location,
            popup=node_name,
            icon=folium.Icon(color=node_color, icon=node_icon, prefix='fa'),
            draggable = True
        ).add_to(m)

    output_file = "map.html"
    m.save(output_file)
    webbrowser.open(output_file, new=2)  # open in new tab

    return m._repr_html_()


if __name__ == '__main__':

    display_map = True
    if display_map:
        instance_name = 'inst_realistic'
        instance_data = load_json_instance('./instances', instance_name + '.json')
        nodes = pd.DataFrame.from_dict(instance_data['nodes_info'])

        new_row_depot    = {'node_name': 'depot_example', 'lat': 41.75190067908858, 'long': 12.236004819472123, 'node_type': 'depot' ,   'node': 60 }
        new_row_facility = {'node_name': 'depot_example', 'lat': 41.75190067908858, 'long': 12.236004819472123, 'node_type': 'facility', 'node': 61}
        new_row_client   = {'node_name': 'depot_example', 'lat': 41.75190067908858, 'long': 12.236004819472123, 'node_type': 'client',   'node': 62 }
        nodes.loc[len(nodes)] = new_row_depot
        nodes.loc[len(nodes)] = new_row_facility
        nodes.loc[len(nodes)] = new_row_client




        nodes.lat = pd.to_numeric(nodes.lat)
        nodes.long = pd.to_numeric(nodes.long)
        start = time.time()
        print('start map creation')
        m = get_folium_map(nodes)
        print('map created in ' + str(time.time() - start))