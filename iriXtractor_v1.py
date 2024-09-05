"""Convert iri.csv file to json and extracts IMEI related information as well as location."""
import csv
import questionary
import folium
from folium.plugins import ScrollZoomToggler, HeatMap, Draw
from geopy.distance import geodesic
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from argparse import ArgumentParser, RawTextHelpFormatter
from zoneinfo import ZoneInfo
from questionary import Style
from rich import print as rprint
from rich.panel import Panel
from textwrap import dedent
from mydebug import timer

# TODO: handle sigint.

# Questionary styling.
custom_style = Style([
    # ('qmark', 'fg:#673ab7 bold'),       # token in front of the question
    # ('question', 'bold'),               # question text
    # ('answer', 'fg:#f44336 bold'),      # submitted answer text behind the question
    ('pointer', 'fg:#f8b301 bold'),     # pointer used in select and checkbox prompts
    ('highlighted', 'fg:#f8b301 bold'), # pointed-at choice in select and checkbox prompts
    # ('selected', 'fg:#cc5454'),         # style for a selected item of a checkbox
    # ('separator', 'fg:#cc5454'),        # separator in lists
    # ('instruction', ''),                # user instructions for select, rawselect, checkbox
    ('text', 'fg:#f8b301'),                       # plain text
    # ('disabled', 'fg:#858585 italic')   # disabled choices for select and checkbox prompts
])

curdir = os.getcwd()
def get_path_to_iri():
    '''Get path to iri files from user.'''
    zip_path = questionary.path("Indicate compressed iri file:").unsafe_ask()
    files_at_path = os.listdir(zip_path)
    print(files_at_path)
    files = [file for file in files_at_path]
    zip_file = questionary.select("Choose a file", choices=files, style=custom_style).unsafe_ask()

    try:
        shutil.copy2(zip_path + '/' + zip_file, curdir)
        rprint(Panel.fit(f"{zip_file} copied to currrent directory.", border_style='cyan'))
    except Exception as exc:
        rprint(f"Cannot copy {zip_file}:\n{exc}")

    return zip_file


def unzip_file(zip_file):
    '''Unzip file.'''
    with zipfile.ZipFile(zip_file, 'r') as myzip:
        myzip.extractall(curdir)
    os.remove(zip_file)


def find_iri_csv():
    '''Find the csv iri file.'''
    current_dir = os.getcwd()
    for root, _, files in os.walk(current_dir):
        for file in files:
            if file.startswith('iri') and file.endswith('csv'):
                f_path = os.path.join(root, file)
                new_name = '/iri.csv'
                try:
                    shutil.copy2(f_path, current_dir + new_name)
                except shutil.SameFileError:
                    pass


iriFile = "iri.csv"
iriJsonFile = "iri.json"

def csv_to_json(csv_f, js_file):
    '''Transpose "normalized" field from iri.csv to json format.'''
    json_data = []
    with open(csv_f, newline='\n') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=';')
        next(csv_reader) # Skip headers.

        for row in csv_reader:
            raw_field = row[8] # Field: normalized.
            try:
                json_object = json.loads(raw_field)
                json_data.append(json_object)
            except json.JSONDecodeError:
                print(f"Error decoding json: {raw_field}")

    json_output = json.dumps(json_data, indent=2)
    with open(js_file, 'w') as wf:
        wf.write(json_output)

def json_to_dataframe(jsfile):
    '''Load json file into dataframe'''
    df = pd.read_json(jsfile)
    # Takes only 14-digit number as n15 is check-digit.
    df['imei'] = df['imei'].astype(str).str[:14].astype('Int64')
    df.drop(['aPN'], axis=1, inplace=True) # Does not influence output.
    print(df)

    # Get counts per unique imei value.
    df_imei_counts = df['imei'].value_counts()
    rprint(df_imei_counts)
    df_imei_counts.to_csv('imei_counts.csv')
    print()

    # df['counts'] = df['imei'].value_counts()
    # print(df.counts)

    # Get liids per unique imei value.
    # Ensure that imei is related to unique liid.
    df_imei_liid = df.groupby('imei')['liid'].unique()
    rprint(df_imei_liid)
    df_imei_liid.to_csv('imei_liid.csv')


class Cell():
    def __init__(self, id, imei, latitude, longitude, azimuth, ts, networkElementId, first_seen, last_seen):
        self.id = id
        self.imei = imei
        self.latitude = latitude
        self.longitude = longitude
        self.azimuth = [azimuth]
        self.ts = ts
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.networkElementId = networkElementId
        self.count = 1

    def increment_cell_count(self):
        self.count += 1

    def append_azimuth(self, azimuth):
        self.azimuth.append(azimuth)

    def update_time_seen(self, first_seen, last_seen):
        self.first_seen = first_seen # new.
        self.last_seen = last_seen # new.


class NetworkElementID():
    def __init__(self, neteid):
        self.neteid = neteid
        self.count = 1

    def increment_neteid_count(self):
        self.count += 1


def json_parser(js_file):
    '''Parse json file and get cell location.'''
    with open(js_file, 'r', encoding='utf-8') as rf:
        jsFile = json.load(rf)

    # The content of json file is a list of dictionaries.
    cell_dic = {}
    lat = ''
    id = ''
    imei = ''
    long = ''
    azimuth = ''
    time_stamp = ''
    network_element_id = ''
    firstSeen = ''
    lastSeen = ''

    # TODO: continue modifying as in celloc.

    network_eid_dic = {}

    err_keys = 0

    for item in jsFile:
        try:
            item['aPN']
            # counter += 1
        except KeyError:
            continue
        else:
            try:
                id = item['cell']['id']
                imei = item['imei'][:-1]
                lat = item['location']['wgs84']['latitude']
                long = item['location']['wgs84']['longitude']
                azimuth = item['location']['azimuth']
                time_stamp = item['cell']['timestamp']
                network_element_id = item['networkElementId']
            except KeyError:
                err_keys += 1 # Not useful.
                continue

        # WARNING: the next time conversion is working.
        # Un-commenting timestamp will convert all time to CH timezone.
        # This could lead to errors if target's tz in another tz.
        timestamp = datetime.strptime(time_stamp[:-1], '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=ZoneInfo('UTC'))
        # timestamp = timestamp.astimezone(ZoneInfo('Europe/Zurich'))
        time_stamp = timestamp.replace(tzinfo=timezone.utc)

        # INFO: to remove when script tested successfully over time.
        # To get unix timestamp.
        # unix_time = datetime.strptime(time_stamp[:-1], '%Y-%m-%dT%H:%M:%S.%f')
        # unix_time = unix_time.timestamp()
        # print(f"{unix_time = }")
        # time_stamp = unix_time

        # Get cell related data.
        if id in cell_dic:
            cell_dic[id].increment_cell_count()
            if azimuth not in cell_dic[id].azimuth:
                cell_dic[id].append_azimuth(azimuth)

            # Assess first and last seen.
            if cell_dic[id].first_seen == '':
                firstSeen = time_stamp
                lastSeen = time_stamp
                cell_dic[id].update_time_seen(firstSeen, lastSeen)
            elif time_stamp > cell_dic[id].last_seen:
                lastSeen = time_stamp
                cell_dic[id].update_time_seen(firstSeen, lastSeen)
            elif time_stamp < cell_dic[id].first_seen:
                firstSeen = time_stamp
                cell_dic[id].update_time_seen(firstSeen, lastSeen)
        else:
            firstSeen = time_stamp
            lastSeen = time_stamp
            cell = Cell(id, imei, lat, long, azimuth, time_stamp, network_element_id, firstSeen, lastSeen)
            cell_dic[id] = cell

        # Get networkElementId (ip address which connects to a cell).
        if network_element_id in network_eid_dic:
            network_eid_dic[network_element_id].increment_neteid_count()
        else:
            neteid = NetworkElementID(network_element_id)
            network_eid_dic[network_element_id] = neteid

    # Build cells data and create dataframe.
    cell_data = []
    for _, val in cell_dic.items():
        # if val.id != '': # Do not take into account empty values, preferable then pd.dropna.
        if val.id != '' and val.latitude != '' and val.longitude != '':
            try:
                cell_data.append({
                    'Cell_id': val.id,
                    'IMEI': val.imei,
                    'Counts': val.count,
                    'NetworkElementId': val.networkElementId,
                    'lat': val.latitude,
                    'long': val.longitude,
                    'azimuth': val.azimuth,
                    'ts': val.ts,
                    'First_seen': val.first_seen.strftime('%d.%m.%Y %H:%M:%S %Z'), # new.
                    'Last_seen': val.last_seen.strftime('%d.%m.%Y %H:%M:%S %Z') # new.
                })
            except Exception as exc:
                db(f"{val.first_seen}", 'red')
                db(f"{val.last_seen}", 'blue')
                print(f"Error: {exc}")

    if len(cell_data) > 0:
        celldf = pd.DataFrame(cell_data)
        # Folium heatmap requires weight from 0 to 1.
        max = celldf['Counts'].max()
        zeros = int(len(str(max))) # e.g.: int(1234) = str(4).
        divider = 10**zeros # e.g.: for 1234 => 10000.

        celldf['weight'] = (celldf['Counts'] / divider)
    else:
        # Create an empty df.
        celldf = pd.DataFrame()

    # Build NetworkElementId data (ip addresses) and create dataframe.
    neteid_data = []
    for _, val in network_eid_dic.items():
        neteid_data.append({
            'NetworkElementId': val.neteid,
            'Counts': val.count
        })

    neteid_df = pd.DataFrame(neteid_data)
    neteid_df.replace('', np.nan, inplace=True)
    neteid_df.dropna(inplace=True)
    neteid_df.sort_values(['Counts'], ascending=False, inplace=True)

    return celldf, neteid_df


def add_azimuth_line(map_object, start_lat, start_lon, azimuth, length_km, popup_content):
    '''Add azimuth line to each cell using geodesic calculation.'''
    cell_location = (start_lat, start_lon)
    end_point = geodesic(kilometers=length_km).destination(cell_location, azimuth)
    end_lat, end_lon = end_point.latitude, end_point.longitude
    popup = folium.Popup(popup_content, max_width=250)
    folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)],
                    weight=6, opacity=0.4, color='#08F7FE', popup=popup).add_to(map_object)


# TODO: modify like in celloc.py.
def transpose_cells_on_map(js_file):
    '''Transpose cell tower coordinates on map.'''
    celldf, neteid_df = json_parser(js_file)

    # Center map on Switzerland centre position.
    m = folium.Map(location=[46.8182, 8.2275], zoom_start=8, tiles="Cartodb dark_matter")
    # m = folium.Map(location=[46.9545639, 7.3123655], zoom_start=8, tiles="Cartodb dark_matter")

    # Block scroll zoom by default.
    scrollonoff = ScrollZoomToggler()
    m.add_child(scrollonoff)

    # Allow to add marker with click, remove with double-click.
    # clickMarker = folium.ClickForMarker()
    # m.add_child(clickMarker)

    # Allow to draw shapes and add markers.
    Draw(export=False).add_to(m)

    # Group_1: id.orig_h.
    heat = folium.FeatureGroup("Cell HeatMap").add_to(m)
    cell_azimuth = folium.FeatureGroup("Cell Azimuth", show=True).add_to(m)

    # Get each cell tower coordinates, longitude: x-axis, latitude: y-axis.
    data = []
    for _, row in celldf.iterrows():
        # Get cell location and cell counter.
        cell_data = [row['lat'], row['long'], row['weight']]
        data.append(cell_data)

        # Draw azimuth line.
        latitude = row['lat']
        longitude = row['long']
        km = 2.5

        popup_content = f"""
                <strong>Cell id: {row['Cell_id']}</strong><br>
                Latitude: {latitude}<br>
                Longitude: {longitude}<br>
                Azimuth: {row['azimuth']}<br>
                <br>
                NetworkElementId: {row['NetworkElementId']}<br>
                IMEI: {row['IMEI']}<br>
                First seen: {row['First_seen']}<br>
                Last seen: {row['Last_seen']}<br>
                Counts: {row['Counts']}
                """

        for azimuth in row['azimuth']:
            add_azimuth_line(cell_azimuth, latitude, longitude, azimuth, km, popup_content)

    # Default, radius=25, blur=15.
    HeatMap(data).add_to(heat)

    folium.LayerControl().add_to(m)

    map_file = 'cells.html'

    m.save(map_file)

    return map_file, neteid_df


@timer
def main():
    # zipFile = get_path_to_iri()
    # unzip_file(zipFile)
    find_iri_csv()
    csv_to_json(iriFile, iriJsonFile)
    json_to_dataframe(iriJsonFile)
    transpose_cells_on_map('iri.json')


# TODO: add an option to erase non-necessary files.
if __name__ == "__main__":
    argMessage = dedent('''\
                        Script that format iri.csv to json then parses the data to retrieve IMEI.''')
    parser = ArgumentParser(description=argMessage, formatter_class=RawTextHelpFormatter)
    parser.usage = ("Takes path to iri file as argument")
    args = parser.parse_args()

    main()

