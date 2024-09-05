"""Convert iri.csv file to json and extracts IMEI related information as well as location."""

import csv
import questionary
import folium
from folium.plugins import ScrollZoomToggler, HeatMap, Draw, MarkerCluster
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


def json_to_dataframe(js_file):
    '''
    Load json file into a dataframe, flatten its structure and return df.

    Handle both Swiss and non-Swiss cell ids.
    '''

    # Load json file to dataframe.
    df = pd.read_json(js_file)

    # Takes only 14-digit number as n15 is check-digit.
    df['imei'] = df['imei'].astype(str).str[:14].astype('Int64')

    # Check the content of columns for dictionaries.
    hasdic = []
    cols = df.columns

    def identify_column_content_type(column, col_name):
        '''Check if column contains dictionary values to flatten.'''
        for item in column:
            if isinstance(item, dict):
                hasdic.append(col_name)
                return

    for col in cols:
        identify_column_content_type(df[col], col)

    # rprint(f"Needs flattening:\n[green]{hasdic}[/]")

    # If 'location' column not found, it means only non-Swiss cells found.
    # The column is created with np.nan values to get same location format.
    isloc = 'location'
    if isloc not in hasdic:
        hasdic.append(isloc)
        nan = np.nan
        data = [{
                "location": {
                    "lv03": {"e": nan, "n": nan},
                    "lv95": {"e": nan, "n": nan},
                    "wgs84": {"latitude": nan, "longitude": nan},
                    "azimuth": nan}
                }]
        df['location'] = pd.DataFrame(data)

        # Prevent flattening column "addtionalProperties" (redundant data).
        # Only found in non-Swiss data.
        try:
            hasdic.remove('additionalProperties')
        except Exception as exc:
            rprint(f"Exception: [red]{exc}[/]")

    # Flattening columns.
    flattened_dfs = {}
    for col in hasdic:
        try:
            # Split columns.
            flattened_df = pd.json_normalize(df[col])
            # Rename colums.
            flattened_df.columns = [f'{col}_{subcol}' for subcol in flattened_df.columns]
            flattened_dfs[col] = flattened_df
        except Exception as exc:
            rprint(f"[red]{exc}[/]")

    # Drop the original column in original df and concat new columns.
    df = df.drop(hasdic, axis=1)
    for col in hasdic:
        df = pd.concat([df, flattened_dfs[col]], axis=1)

    # Remove empty cell_id.
    df = df.dropna(subset=['cell_id'])

    # Split df to Swiss and non-Swiss cell_id dfs.
    filt = df['cell_id'].str.startswith('228')
    sdf = df[filt]
    nosdf = df[~filt]

    # TEST: simulate getting location from db.
    # Foreign cell_ids can have both formats (MCC-MNC-LAC-ID or MCC-MNC-ID).
    # To take into consideration later.
    # 230-01-107776267
    # 230-01-14418-20977

    cities_dic = {'London': [+51.5002, -0.1262],
                  'Oslo': [+59.9138, +10.7387],
                  'Rome': [+41.8955, +12.4823],
                  'Washington': [+38.8921, -77.0241],
                  'Pretoria': [-25.7463, +28.1876],
                  'Canberra': [-35.2820, +149.1286],
                  'Wales': [+65.620, -168.1336],
                  'McMurdo': [-77.8401, +166.6424],
                  'Svalbard': [+77.2408, +12.1280]}

    cities = ['London', 'Oslo', 'Rome', 'Washington', 'Pretoria', 'Canberra', 'Wales', 'McMurdo', 'Svalbard']

    cellids = nosdf['cell_id']
    import random
    for cell in cellids:
        filt = (nosdf['cell_id'] == cell)
        city = random.choice(cities)
        latitude = cities_dic[city][0]
        longitude = cities_dic[city][1]
        azimuth = random.randrange(0,360,10)

        nosdf.loc[filt, 'location_wgs84.latitude'] = latitude
        nosdf.loc[filt, 'location_wgs84.longitude'] = longitude
        nosdf.loc[filt, 'location_azimuth'] = azimuth

    # Create the final dataframe.
    df = pd.concat([sdf, nosdf], axis=0)
    df.to_csv('s1_df.tsv', sep='\t', index=False)

    return df


class Cell():
    def __init__(self, id, imei, latitude, longitude, azimuth, networkElementId, first_seen, last_seen, count):
        self.id = id
        self.imei = imei
        self.latitude = latitude
        self.longitude = longitude
        self.azimuth = [azimuth]
        self.networkElementId = networkElementId
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.count = count

    def increment_cell_count(self):
        self.count += 1

    def append_azimuth(self, azimuth):
        self.azimuth.append(azimuth)

    def update_time_seen(self, first_seen, last_seen):
        self.first_seen = first_seen
        self.last_seen = last_seen


def dataframe_parser(dataframe):
    '''Parse the dataframe to get cell location related data only.'''

    # if dataframe.empty:
    #     celldf = pd.DataFrame()
    #     neteid_df = pd.DataFrame()
    #     return celldf, neteid_df # neteid_df to be created.

    df = dataframe[['cell_id', 'imei', 'location_wgs84.latitude', 'location_wgs84.longitude',
             'location_azimuth', 'cell_timestamp', 'networkElementId']]
             # 'location_azimuth', 'cell_timestamp', 'ts', 'networkElementId']]

    # Convert timestamp to datetime, this will be beneficial later.
    pd.set_option('mode.chained_assignment', None)
    df['cell_timestamp'] = pd.to_datetime(df.loc[:, 'cell_timestamp'])

    # DO NOT REMOVE!
    df = df.dropna()

    # Allow parsing data with visidata.
    df.to_csv('s2_cell_data.tsv', sep='\t', index=False)

    # Get unique cells.
    cells = df['cell_id'].unique()

    cell_dic = {}
    firstSeen = ''
    lastSeen = ''

    for cell in cells:
        filt = (df['cell_id'] == cell)
        id = cell
        imei = df[filt]['imei'].unique()[0]
        lat = df[filt]['location_wgs84.latitude'].unique()[0]
        long = df[filt]['location_wgs84.longitude'].unique()[0]
        azimuth = df[filt]['location_azimuth'].unique()
        firstSeen = df[filt]['cell_timestamp'].min()
        lastSeen = df[filt]['cell_timestamp'].max()
        network_element_id = df[filt]['networkElementId'].unique()[0]
        counts = df[filt].value_counts().sum()

        cell = Cell(id, imei, lat, long, azimuth, network_element_id, firstSeen, lastSeen, counts)
        cell_dic[id] = cell

    # Build cells data and create dataframe.
    cell_data = []
    for _, val in cell_dic.items():
        try:
            cell_data.append({
                'Cell_id': val.id,
                'IMEI': val.imei,
                'Counts': val.count,
                'NetworkElementId': val.networkElementId,
                'lat': val.latitude,
                'long': val.longitude,
                'azimuth': val.azimuth,
                'First_seen': val.first_seen,
                'Last_seen': val.last_seen
            })
        except Exception as exc:
            print(f"Error: {exc}")

    # TODO: assess.
    # If condition should not be necessary as if no location, empty df is created.
    if len(cell_data) > 0:
        celldf = pd.DataFrame(cell_data)
        # Folium heatmap requires weight from 0 to 1.
        max = celldf['Counts'].max()
        zeros = int(len(str(max))) # e.g.: int(1234) = str(4).
        divider = 10**zeros # e.g.: for 1234 => 10000.
        celldf['weight'] = (celldf['Counts'] / divider)
        celldf['First_seen'] = celldf['First_seen'].dt.strftime('%d.%m.%Y %H:%M:%S UTC')
        celldf['Last_seen'] = celldf['Last_seen'].dt.strftime('%d.%m.%Y %H:%M:%S UTC')

    else:
        # Create an empty df.
        celldf = pd.DataFrame()

    neteid_df = pd.DataFrame()

    # Allow parsing celldf with visidata.
    celldf.to_csv('s3_celldf.tsv', sep='\t', index=False)

    return celldf, neteid_df


def add_azimuth_line(map_object, start_lat, start_lon, azimuth, length_km, tool_tip):
    '''Add azimuth line to each cell using geodesic calculation.'''
    cell_location = (start_lat, start_lon)
    end_point = geodesic(kilometers=length_km).destination(cell_location, azimuth)
    end_lat, end_lon = end_point.latitude, end_point.longitude
    folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)],
                    weight=5, opacity=0.4, color='#08F7FE', tooltip=tool_tip).add_to(map_object)


def transpose_cells_on_map(dataframe):
    '''Transpose cell tower coordinates on map.'''
    celldf, neteid_df = dataframe_parser(dataframe)


    # Center map on Switzerland centre position.
    m = folium.Map(location=[46.8182, 8.2275], zoom_start=2, tiles="Cartodb dark_matter")

    # Block scroll zoom by default.
    scrollonoff = ScrollZoomToggler()
    m.add_child(scrollonoff)

    # Allow to draw shapes and add markers.
    Draw(export=False).add_to(m)

    # Add features on upper right handside corner.
    heat = folium.FeatureGroup("Cell HeatMap", show=True).add_to(m)
    cell_azimuth = folium.FeatureGroup("Cell Azimuth", show=False).add_to(m)
    cell_data = folium.FeatureGroup("Cell Data", show=False).add_to(m)
    mCluster = MarkerCluster().add_to(cell_data)

    # Create popup content for cell_data (iterrows: index, row).
    for _, row in celldf.iterrows():
        popup_content = f"""
                        <strong>Cell id: {row['Cell_id']}</strong><br>
                        Latitude: {row['lat']}<br>
                        Longitude: {row['long']}<br>
                        Azimuth: {int(row['azimuth'][0][0])}<br>
                        <br>
                        NetworkElementId: {row['NetworkElementId']}<br>
                        IMEI: {row['IMEI']}<br>
                        First seen: {row['First_seen']}<br>
                        Last seen: {row['Last_seen']}<br>
                        Counts: {row['Counts']}
                        """

        # popup: on click, tooltip: on hover.
        folium.Marker(location=[row['lat'], row['long']],
                      popup=folium.Popup(popup_content, max_width=250),
                      tooltip=f"{row['Cell_id']}")\
                      .add_to(mCluster)

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

        # Add each azimuth per cell.
        for azimuth in row['azimuth']:
            tooltipTag = int(azimuth[0])
            add_azimuth_line(cell_azimuth, latitude, longitude, azimuth, km, tooltipTag)

    # db(data)
    # Default, radius=25, blur=15.
    HeatMap(data).add_to(heat)

    folium.LayerControl().add_to(m)

    map_file = 'cells.html'

    m.save(map_file)

    return map_file, neteid_df


@timer
def main():
    zipFile = get_path_to_iri()
    unzip_file(zipFile)
    find_iri_csv()
    csv_to_json(iriFile, iriJsonFile)
    dframe = json_to_dataframe(iriJsonFile)
    transpose_cells_on_map(dframe)


# TODO: add an option to erase non-necessary files.
if __name__ == "__main__":
    argMessage = dedent('''\
                        Script that format iri.csv to json then parses the data to retrieve IMEI.''')
    parser = ArgumentParser(description=argMessage, formatter_class=RawTextHelpFormatter)
    parser.usage = ("Takes path to iri file as argument")
    args = parser.parse_args()

    main()

