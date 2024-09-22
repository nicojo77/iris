"""Convert iri.csv file to json and extracts IMEI related information as well as locations."""

import csv
import folium
import questionary
from folium.plugins import ScrollZoomToggler, HeatMap, Draw, MarkerCluster
from geopy.distance import geodesic
import json
import mobile_codes
import numpy as np
import os
import pandas as pd
import requests
import shutil
import sys
import time
import zipfile
# from datetime import datetime, timezone
from argparse import ArgumentParser, RawTextHelpFormatter
# from zoneinfo import ZoneInfo
from questionary import Style
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from textwrap import dedent
import constants

from mydebug import timer
from mydebug import db


console = Console()

# TODO: 
# Handle sigint.
# Assess what tsv files should be created, certainly too much currently.
# Assess if non-Swiss cells shoulb be UTC or Swiss-Time, check with emco.
# Embed iris.py into capcap.py


# TODO:
# Find a way to protect the file against deletion.
IFC_CELLT_DB = '/home/anon/Documents/git/pythonScripts/iris/IFC_CELLT_DB.parquet'

# Counters for statistics.
ifc_localised = 0
ifc_checked = 0
google_localised = 0
google_checked = 0
combain_localised = 0
combain_checked = 0

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

    Return:
    base_df, nominal dataframe.
    tot_cells_dic, dictionary.
    '''

    # Load json file to dataframe.
    df = pd.read_json(js_file)
    df['imei'] = df['imei'].astype('Int64', copy=False)
    df.to_csv('df.tsv', sep='\t')

    # WARNING: dropna() appears to crash the structure.
    # df = df.dropna()
    # df.to_csv('df_dropna.tsv', sep='\t')

    # Takes only 14-digit number as n15 is check-digit.
    df.dropna(subset=['imei'], inplace=True)
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
    # ['domainId', 'targetIPAddress', 'correlationNumber', 'area', 'cell', 'location', 'additionalProperties']

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
    except ValueError:
        pass
    except Exception as exc:
        rprint(f"[red]Exception: [/]{exc}")

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
    base_df = df.dropna(subset=['cell_id'])

    # Split column 'cell_id' (dtypes: object) into values' specific element.
    base_df = base_df.copy() # Ensure working on copy.
    base_df['mcc'] = base_df['cell_id'].apply(lambda x: x.split('-')[0])
    base_df['mnc'] = base_df['cell_id'].apply(lambda x: x.split('-')[1])
    base_df['lac'] = base_df.apply(
        lambda row: row['cell_id'].split('-')[2] if row['cell_idtype'] in ['CGI', 'SAI']
        else (row['area_id'].split('-')[2] if row['cell_idtype'] == 'ECGI' else np.nan),
        axis=1)
    base_df['cid'] = base_df['cell_id'].apply(lambda x: x.split('-')[-1])
    # base_df['ecid_short'] = base_df.apply(lambda row: (int(row['cid']) % 65536) if row['cell_idtype'] == 'ECGI' else np.nan, axis=1).astype('Int64')

    base_df.to_csv('base_df.tsv', sep='\t')

    # Get the initial counts for each cell.
    # This never changes and only used in mcc_checker().
    base_df['mcc'] = base_df['mcc'].astype(str)
    mcc_list = base_df['mcc'].unique()
    tot_cells_dic = {}
    for mcc in mcc_list:
        filt = (base_df['mcc'] == mcc)
        tot_cells = base_df[filt]['cell_id'].count()
        tot_cells_dic[mcc] = tot_cells

    return base_df,tot_cells_dic


    # TODO: so far matching ecid with openCellID db works.
    # (capcap310) anon@X1:~/Desktop/cases/odyssee_iri$
    # (capcap310) anon@X1:~/Desktop/cases/iri$
    # (capcap310) anon@X1:~/Desktop/cases/irimaxi$
    # In odyssee_iri you can use cmdlog.vdj to directly get to non-identified cells so far


def split_process_concat_dataframe(dataframe):
    '''
    Split base dataframe, get locations and concat again.

    sdf: Swiss Dataframe, no processing as location already known.
    nosdf: non-Swiss Dataframe, crosscheck against cell-tower db, and apis.

    This function is launcher of: check_cell_towers(data)

    Return:
    df, un-filtered dataframe (swiss and foreign cell-towers).
    data, data format with [cell_id, lat, lon, ts].
    '''
    # Remove un-wanted columns dynamically with sets.
    actual_cols = set(dataframe.columns)
    wanted_cols = set([
                    'imei',
                    'imsi',
                    'liid',
                    'iriTimestamp',
                    'targetAddress',
                    'networkElementId',
                    'area_id',
                    'area_idtype',
                    'cell_id',
                    'cell_idtype',
                    'cell_timestamp',
                    'location_azimuth',
                    'location_wgs84.latitude',
                    'location_wgs84.longitude',
                    'targetIPAddress_IPv4Address',
                    'targetIPAddress_IPv6Address',
                    'mcc',
                    'mnc',
                    'lac',
                    'cid',
                    'ecid_short',
                    'area'
                    ])

    to_remove_cols = actual_cols.difference(wanted_cols)
    df = dataframe.drop(list(to_remove_cols), axis=1)
    df.to_csv('unprocessed_df.tsv', sep='\t', index=False)
    init_df = df

    # _: OPENCELLID:
    #
    # PART 1

    # Get un-localised cells and remove duplicates on cell_id.
    df = df[df['location_wgs84.latitude'].isna()]
    df = df.drop_duplicates(subset=['cell_id'])
    df = df.astype({'mcc': 'Int64', 'mnc': 'Int8', 'lac': 'Int64', 'cid': 'Int64'})

    openCellID = '/home/anon/Desktop/it_stuff/openCellID/cell_towers.parquet'
    ocid_df = pd.read_parquet(openCellID, columns=['mcc', 'net', 'area', 'cell', 'lon', 'lat'])

    df = df.merge(ocid_df[['mcc', 'net', 'area', 'cell', 'lat', 'lon']],
                                    left_on=['mcc', 'mnc', 'lac', 'cid'],
                                    right_on=['mcc', 'net', 'area', 'cell'],
                                    how='left')

    # Ensure proper handling of NaN values.
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    # Populate coordinates (lat, lon) to empty location_wgs84 when matches occur.
    df['location_wgs84.latitude'] = df['location_wgs84.latitude'].fillna(df['lat'])
    df['location_wgs84.longitude'] = df['location_wgs84.longitude'].fillna(df['lon'])
    df.drop(['lat', 'lon', 'cell', 'area', 'net'], axis=1)

    # _: IFC_DB, GOOGLE AND COMBAIN APIS.
    #
    # PART 2

    # Get rid off every cell-towers identified by openCellID.
    nosdf = df
    nosdf = nosdf[nosdf['location_wgs84.latitude'].isna()]


    # Create a set with (cell_id, mcc, mnc, lac and cid).
    # Set will get rid off duplicates automatically.
    data = set()
    for _, row in nosdf.iterrows():
        cell_data = (row['cell_id'], row['mcc'], row['mnc'], row['lac'], row['cid'])
        data.add(cell_data)

    # TEST: LIMIT DATA TO N NUMBER OF CELLS.
    # data = sorted(data)
    # data = list(data)[:10]

    # Perform the checks in ifc_db, google and combain apis.
    localised, newdf = check_cell_towers(data)

    # Merge coordinates found in check_cell_towers()
    new_df = init_df.merge(newdf[['cell_id', 'lat', 'lon']], left_on=['cell_id'], right_on=['cell_id'], how='left')

    # Ensure proper handling of NaN values.
    new_df['lat'] = pd.to_numeric(new_df['lat'], errors='coerce')
    new_df['lon'] = pd.to_numeric(new_df['lon'], errors='coerce')

    # Populate coordinates (lat, lon) to empty location_wgs84 when matches occur.
    # Drop un-necessary columns.
    new_df['location_wgs84.latitude'] = new_df['location_wgs84.latitude'].fillna(new_df['lat'])
    new_df['location_wgs84.longitude'] = new_df['location_wgs84.longitude'].fillna(new_df['lon'])
    final_df = new_df.drop(['lat', 'lon'], axis=1)
    final_df.to_csv('findf.tsv', sep='\t', index=False)

    return localised, init_df, final_df


def api_requester(api, url_, cellTower_data):
    '''
    Handle POST requests process on Cell-Towers db and apis.

    Called by check_cell_towers().

    Parameters:
    api: str, name of api being checked
    cellTower_data: dict, contains unique cell tower data
    max_retries: int.
    '''

    global launch_google_api
    global launch_combain_api

    headers = {
        "Content-Type": "application/json"
    }
    request_data = {
        "considerIp": False,
        "cellTowers": cellTower_data
    }

    current_delay = 0.1 # Set initial retry delay to 100ms.
    max_delay = 3 # Set maximum retry delay to 3s (5 attempts).
    while True:
        try:
            response = requests.post(url_, headers=headers, json=request_data)
            response.raise_for_status() # Raise an exception for 4xx/5xx errors.
            return  response.json() # If successful, return the result.

        except requests.exceptions.ConnectionError:
            rprint("[red]Network error: unable to connect to Internet.")
            sys.exit(1)

        except requests.exceptions.HTTPError:
            status_code = response.status_code
            # error_message = response.json().get("error", {}).get("message", "")
            # print(f"Error: {status_code} - {error_message}")

            # Cell tower not found, no point retrying.
            if status_code == 404:
                if api == 'google':
                    launch_combain_api = True
                break

            # Api issues (limit or key), no point retrying.
            if status_code in (400, 403):
                if api == 'google':
                    rprint(f"[red]Something went wrong with Google api![/]")
                    error_google_api = True
                elif api == 'combain':
                    rprint(f"[red]Something went wrong with Combain api![/]")
                    error_combain_api = True
                break

        # Too many attempts, meaning something is wrong with internet connection.
        # If both Google and Combain encounter issues, script should stop.
        if current_delay > max_delay:
            if api == 'google':
                rprint(f"[red]Google api not reachable! Continuing with Combain only[/]")
                launch_google_api = False
            elif api == 'combain':
                rprint(f"[red]Combain api not reachable either![/]")
                launch_combain_api = False
            raise Exception("Too many retry attempts.")

        # For other errors (like network issues), retry with exponential backoff.
        print(f"Waiting {current_delay}s before retrying.")
        time.sleep(current_delay)
        current_delay *= 2 # Increase delay at each retrial.

    return None


def check_cell_towers(cellTower_dataList_):
    '''
    Take cell tower data list from split_process_concat_dataframe().
    Determine what checks needs performing (ifc_db or apis).
    Feed api_requester() with a list of cell-towers, parse answers.

    Parameter: cellTower_dataList_ (list), list of cell-towers to be checked.

    Return:
    localised (list): data format of cell-towers (cell_id, lat, lon, ts).
    df (pandas df): newly identified cell-towers.
    '''

    ctdl = cellTower_dataList_

    global ifc_localised
    global google_localised
    global combain_localised
    global ifc_checked
    global google_checked
    global combain_checked
    global launch_google_api
    global launch_combain_api

    # Load ifc3 own database.
    if os.path.isfile(IFC_CELLT_DB):
        # db(f"{IFC_CELLT_DB} does exist.", colour='green')
        ifc_df = pd.read_parquet(IFC_CELLT_DB)
        # Filter on current year.
        # ts_cut = int(time.time()) - 20 # test: get rid of files older than 20 seconds. 
        ts_cut = int(time.time()) - 31536000 # 1 year.
        filt = (ifc_df['ts'] > ts_cut)
        ifc_df = ifc_df[filt]

    # Database does not exist, create empty template with columns.
    else:
        # db(f"{IFC_CELLT_DB} does not exist.", colour='red')
        cols = ['cell_id', 'lat', 'lon', 'ts']
        ifc_df = pd.DataFrame(columns=cols)
        ifc_df.astype({'cell_id': str, 'lat': 'Float64', 'lon': 'Float64', 'ts': 'Int64'})

    localised = []
    i = 0
    length_ctdl = len(ctdl)

    while i < length_ctdl:
        # Determine if google and combain works properly.
        # False: errors 400 or 403.
        error_google_api = False
        error_combain_api = False

        # Google and Combain apis are checked only if cell tower not in IFC_db.
        launch_google_api = False
        launch_combain_api = False

        cellTowerData = [
            {
                "mobileCountryCode": list(ctdl)[i][1],
                "mobileNetworkCode": list(ctdl)[i][2],
                "locationAreaCode": list(ctdl)[i][3],
                "cellId": list(ctdl)[i][4]
            }
        ]

        # Check if cell tower in ifc db.
        cell_id = str(list(ctdl)[i][0])
        if cell_id in ifc_df['cell_id'].values:
            # db(f"{cell_id} in ifc_df", colour='green')
            filt = (ifc_df['cell_id'] == cell_id)
            lat = (ifc_df[filt]['lat'].values)[-1]
            lon = (ifc_df[filt]['lon'].values)[-1]
            ts = (ifc_df[filt]['ts'].values)[-1]
            # db(f"{cell_id = }\n{lat = }\n{lon = }\n{ts = }")
            # db(f"{type(cell_id)}\n{type(lat)}\n{type(lon)}\n{type(ts)}")
            localised.append([cell_id, lat, lon, ts])
            ifc_localised += 1
            ifc_checked += 1
        else:
            # db(f"{cell_id} not in ifc_df", colour='red')
            launch_google_api = True

        # Google Api.
        if launch_google_api and not error_google_api:
            GOOGLE_API_KEY = constants.GOOGLE_API_KEY
            url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
            result = api_requester('google', url, cellTowerData)
            google_checked += 1
            if result:
                lat = result['location']['lat']
                lon = result['location']['lng']
                ts = int(time.time())
                localised.append([list(ctdl)[i][0], lat, lon, ts])
                google_localised += 1

        # Combain api.
        if launch_combain_api and not error_combain_api:
            COMBAIN_API_KEY = constants.COMBAIN_API_KEY
            url = f"https://apiv2.combain.com?key={COMBAIN_API_KEY}"
            result = api_requester('combain', url, cellTowerData)
            combain_checked += 1
            if result:
                lat = result['location']['lat']
                lon = result['location']['lng']
                ts = int(time.time())
                localised.append([list(ctdl)[i][0], lat, lon, ts])
                combain_localised += 1

        i += 1

    # Create dataframe with cell tower locations.
    # Update ifc db with new cell-towers.
    cols = ['cell_id', 'lat', 'lon', 'ts']
    df = pd.DataFrame(localised, columns=cols)
    new_ifc_df = df.combine_first(ifc_df)
    new_ifc_df.to_parquet(IFC_CELLT_DB, index=False)

    return localised, df


class Cell():
    def __init__(self, id, imei, latitude, longitude, azimuth, networkElementId, first_seen, last_seen, count, mcc):
        self.id = id
        self.imei = imei
        self.latitude = latitude
        self.longitude = longitude
        self.azimuth = [azimuth]
        self.networkElementId = networkElementId
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.count = count
        self.mcc = mcc

    def increment_cell_count(self):
        self.count += 1

    def append_azimuth(self, azimuth):
        self.azimuth.append(azimuth)

    def update_time_seen(self, first_seen, last_seen):
        self.first_seen = first_seen
        self.last_seen = last_seen


def dataframe_parser(dataframe):
    '''Parse the dataframe to get cell location related data only.'''

    df = dataframe[['cell_id', 'imei', 'location_wgs84.latitude', 'location_wgs84.longitude',
             'location_azimuth', 'cell_timestamp', 'networkElementId', 'mcc']]

    # Convert timestamp to datetime, this will be beneficial later.
    pd.set_option('mode.chained_assignment', None)
    df['cell_timestamp'] = pd.to_datetime(df.loc[:, 'cell_timestamp'])

    # DO NOT REMOVE!
    df = df.dropna(subset=['location_wgs84.latitude', 'location_wgs84.longitude'], how='any')

    # Fill in empty azimuth with 0, will only apply on non-Swiss cells.
    df['location_azimuth'] = df['location_azimuth'].fillna(0)

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
        mcc = df[filt]['mcc'].unique()[0]
        imei = df[filt]['imei'].unique()[0]
        lat = df[filt]['location_wgs84.latitude'].unique()[0]
        long = df[filt]['location_wgs84.longitude'].unique()[0]
        azimuth = df[filt]['location_azimuth'].unique()
        firstSeen = df[filt]['cell_timestamp'].min()
        lastSeen = df[filt]['cell_timestamp'].max()
        network_element_id = df[filt]['networkElementId'].unique()[0]
        counts = df[filt].value_counts().sum()

        cell = Cell(id, imei, lat, long, azimuth, network_element_id, firstSeen, lastSeen, counts, mcc)
        cell_dic[id] = cell

    # Build cells data and create dataframe.
    cell_data = []
    for _, val in cell_dic.items():
        try:
            cell_data.append({
                'Cell_id': val.id,
                'mcc': val.mcc,
                'IMEI': val.imei,
                'Counts': val.count,
                'NetworkElementId': val.networkElementId, # TODO: assess if necessary?
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
    # m = folium.Map(location=[46.8182, 8.2275], zoom_start=2, tiles="Cartodb positron")
    m = folium.Map(location=[46.8182, 8.2275], zoom_start=2, tiles="Cartodb dark_matter")

    # Block scroll zoom by default.
    # scrollonoff = ScrollZoomToggler()
    # m.add_child(scrollonoff)

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

        # Add azimuth for Swiss cell only.
        if row['Cell_id'].startswith('228'):
            for azimuth_list in row['azimuth']:
                for azimuth in azimuth_list:
                    tooltipTag = int(azimuth)
                    add_azimuth_line(cell_azimuth, latitude, longitude, azimuth, km, tooltipTag)

    # Default, radius=25, blur=15.
    HeatMap(data).add_to(heat)

    folium.LayerControl().add_to(m)

    map_file = 'cells.html'

    m.save(map_file)

    return map_file, neteid_df


# INFO:
# The next snippet shows what should be expected on first run.

# Not localised unique cell-towers:           812
# Cell-towers identified by IFC_CELLT_DB:     0/812 (0.00 %)
# Cell-towers identified by openCellID:       247/812 (30.42 %)
# Cell-towers identified by Google:           200/812 (24.63 %)
# Cell-towers identified by Combain:          18/812 (2.21 %)
# Not localised:                              347 (42.73 %)


def summary(unprocessed_dataframe, localised_):
    '''Display statistics on localisation rate per db or apis.'''

    # Un-localised unique cell-towers:            812
    #   - returns the overall number of unique cells.
    # Cell-towers identified by openCellID:       247/812 (30.42 %)
    #   - returns the number of those cells identified by openCellID.
    #     At this stage, this is n openCellID localised / total unique cells.
    # Cell-towers identified by IFC_CELLT_DB:     565/565 (100.00 %)
    #   - returns the numbers identified in other db, included the cells that have already been found.

    def api_stat(api, x_localised, checked):
        '''Return counts and percentage per api'''
        percentage = 0
        if checked != 0:
            percentage = (x_localised * 100) / checked
            msg = f"Cell-towers identified by {api}:"
            rprint(f"{msg.ljust(44)}{x_localised}/{checked} ({percentage:.2f} %)")

    # Get unique un-localised cells.
    n_unlocalised = unprocessed_dataframe.drop_duplicates(subset=['cell_id'])['location_wgs84.latitude'].isna().sum()
    n_openCId = n_unlocalised - len(localised_)

    msgTot = f"[bold italic green]Un-localised unique cell-towers:[/]"
    rprint(f"{msgTot.ljust(66)}{n_unlocalised}")
    api_stat('openCellID', n_openCId, n_unlocalised)
    api_stat('IFC_CELLT_DB', ifc_localised, n_unlocalised)
    # api_stat('IFC_CELLT_DB', ifc_localised, ifc_checked)
    api_stat('Google', google_localised, google_checked)
    api_stat('Combain', combain_localised, combain_checked)
    print()

    return None


def mcc_checker(unprocessed_dataframe, processed_dataframe, cell_counter_dic):
    '''Statistics on cell-towers and localised ratios.'''

    # Counters for initial cells do not change over time.
    # Values comes from json_to_dataframe().
    totCell = cell_counter_dic

    # Get the list of unique mcc.
    pro_df = processed_dataframe
    pro_df['mcc'] = pro_df['mcc'].astype(str)
    mcc_list = pro_df['mcc'].unique()

    # Create the data structure for statistics.
    data = []
    for mcc in mcc_list:
        filt = (pro_df['mcc'] == mcc)
        country_name = mobile_codes.mcc(mcc)[0].name if mobile_codes.mcc(mcc) else "UNKNOWN"
        unique_cells = pro_df[filt]['cell_id'].nunique()
        # unique_cell_df = pro_df[pro_df['mcc'] == int(mcc)].drop_duplicates(subset=['cell_id'])
        unique_cell_df = pro_df[filt].drop_duplicates(subset=['cell_id'])
        unique_localised = unique_cell_df[unique_cell_df['location_wgs84.latitude'].notna()]['cell_id'].nunique()
        loc_ratios = ((unique_localised * 100) / unique_cells) if unique_cells > 0 else 0
        data.append([mcc, country_name, totCell[mcc] , unique_cells, unique_localised, f"{loc_ratios:.2f}%"])

    cols = ['MCC', 'Country', 'Total_cells', 'Unique_cells', 'Localised', 'LocRatios']
    stat_df = pd.DataFrame(data, columns=cols)
    # Remove index before displaying in terminal.
    stat_df = stat_df.to_string(index=False)

    rprint(Panel.fit(f"{stat_df}", border_style='green', title='MCC Checker', title_align='left'))

    return stat_df


@timer
def main():
    print()
    # zipFile = get_path_to_iri()
    with console.status("[bold][italic][green]Processing iri file...[/]"):
        # unzip_file(zipFile)
        # find_iri_csv()

        csv_to_json(iriFile, iriJsonFile)
        base_df, counterDic = json_to_dataframe(iriJsonFile)
        localised, unprodf, prodf = split_process_concat_dataframe(base_df) # test: dftest
        transpose_cells_on_map(prodf)

    summary(unprodf, localised)
    # mcc_checker(dframe, counterDic)
    mcc_checker(unprodf, prodf, counterDic)


# TODO: add an option to erase non-necessary files.
# Modify argMessage.
if __name__ == "__main__":
    argMessage = dedent('''\
                        Script that format iri.csv to json then parses the data to retrieve IMEI.''')
    parser = ArgumentParser(description=argMessage, formatter_class=RawTextHelpFormatter)
    parser.usage = ("Takes path to iri file as argument")
    args = parser.parse_args()

    main()
