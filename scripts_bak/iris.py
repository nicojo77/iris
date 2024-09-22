"""Convert iri.csv file to json and extracts IMEI related information as well as locations."""

# Both Google and Combain apis share the same error codes.
# 400 = Parse Error / Invalid key.
# 403 = Out of credits.
# 404 = Not found (meaning cell tower not found, api is ok).

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
# if this version is too complicated to handle, a solution would be to get the number of cell checked 
# in openCellID and substract it in summmary (mcc_checker is ok).


# TODO:
# Find a way to protect the file against deletion.

API_CACHED_ONEYEAR = '/home/anon/Documents/git/pythonScripts/iris/API_CACHED_ONEYEAR.parquet'
API_CACHED_ONEDAY = '/home/anon/Documents/git/pythonScripts/iris/API_CACHED_ONEDAY.parquet'

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
    df.to_csv('df.csv')

    # WARNING: dropna() appears to crash the structure.
    # df = df.dropna()
    # df.to_csv('df_dropna.csv')

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

    base_df.to_csv('base_df.csv')

    # Get the initial counts for each cell.
    # This never changes and only used in mcc_checker().
    base_df['mcc'] = base_df['mcc'].astype(str)
    mcc_list = base_df['mcc'].unique()
    tot_cells_dic = {}
    for mcc in mcc_list:
        filt = (base_df['mcc'] == mcc)
        tot_cells = base_df[filt]['cell_id'].count()
        tot_cells_dic[mcc] = tot_cells

    # Remove un-wanted columns dynamically with sets.
    actual_cols = set(base_df.columns)
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
    initial_df = base_df.drop(list(to_remove_cols), axis=1)
    # Remove leading '0' in mnc.
    initial_df['mnc'] = initial_df['mnc'].str.lstrip('0')
    initial_df.to_csv('initial_df.csv', index=False)

    # HACK: test get only small amount of data.
    # mcc = '222'
    # filt = (initial_df['mcc'] == mcc)
    # initial_df = initial_df[filt]
    # db(f"Testing: filtered on mcc {mcc}", colour='orange_red1')

    # Collect statistical data.
    global missing_cells
    missing_coordinates = initial_df.drop_duplicates(subset=['cell_id'])
    missing_cells = missing_coordinates['location_wgs84.latitude'].isna().value_counts()[0]

    return initial_df, tot_cells_dic


def check_cached_oneyear_db(initial_df_):
    '''
    Check unknown cell-towers against API_CACHED_ONEYEAR.

    Parameters:
    initial_df_ (pd df):     initial dataframe properly formatted.

    Return:
    # localised (list):                     cell-towers localised in API_CACHED_ONEYEAR.
    api_cached_oneyear_init_df (pd df):          current API_CACHED_ONEYEAR database.
    api_cached_oneyear_final_df (pd df):    new dataframe with updated coordinates (localised []).
    '''
    # Remark:
    # Be aware that API_CACHED_ONEYEAR also contains data from the other online sources.
    # Take that into account if you try to compare the dataframes.
    # For example, initially you may have 500 un-localised cells.
    # check_cached_oneyear_db() may reveal 400 of them, leaving 100 un-localised.
    # check_opencellid() may return 100 hits out of the 100 un-localised cells by check_cached_oneyear_db.
    # However, those hits could also have been in the first 400 hits.
    #
    # The purpose of API_CACHED_ONEYEAR is only to prevent duplicate requests on paid services.

    init_df = initial_df_

    # Check if API_CACHED_ONEYEAR.parquet exists.
    global ifc_localised
    global ifc_checked

    updated_df = pd.DataFrame()
    if os.path.isfile(API_CACHED_ONEYEAR):
        api_cached_oneyear_init_df = pd.read_parquet(API_CACHED_ONEYEAR)
        # Filter on current year.
        ts_cut = int(time.time()) - 31536000 # 1 year.
        filt = (api_cached_oneyear_init_df['ts'] > ts_cut)
        api_cached_oneyear_init_df = api_cached_oneyear_init_df[filt]

    # API_CACHED_ONEYEAR.parquet does not exist, create empty template with columns.
    else:
        cols = ['cell_id', 'lat', 'lon', 'ts', 'source'] # TEST: 
        # cols = ['cell_id', 'lat', 'lon', 'ts']
        api_cached_oneyear_init_df = pd.DataFrame(columns=cols)
        api_cached_oneyear_init_df.astype({'cell_id': str, 'lat': 'Float64', 'lon': 'Float64', 'ts': 'Int64', 'source': str}) # TEST:
        # api_cached_oneyear_init_df.astype({'cell_id': str, 'lat': 'Float64', 'lon': 'Float64', 'ts': 'Int64'})

    final_df = init_df.merge(api_cached_oneyear_init_df[['cell_id', 'lat', 'lon']], on=['cell_id'], how='left')

    final_df['lat'] = pd.to_numeric(final_df['lat'], errors='coerce')
    final_df['lon'] = pd.to_numeric(final_df['lon'], errors='coerce')

    # Populate coordinates (lat, lon) to empty location_wgs84 when matches occur.
    # Drop un-necessary columns.
    final_df['location_wgs84.latitude'] = final_df['location_wgs84.latitude'].fillna(final_df['lat'])
    final_df['location_wgs84.longitude'] = final_df['location_wgs84.longitude'].fillna(final_df['lon'])
    api_cached_oneyear_final_df = final_df.drop(['lat', 'lon'], axis=1)
    api_cached_oneyear_final_df.to_csv('api_cached_oneyear_final_df.csv', index=False)

    return api_cached_oneyear_init_df, api_cached_oneyear_final_df


def check_opencellid(init_df_, api_cached_oneyear_final_df_):
    '''
    Check unknown cell-towers against openCellID db.

    Parameters:
    init_df_ (pd df):           initial dataframe returned by json_to_dataframe()
    api_cached_oneyear_final_df_ (pd df):   returned by check_cached_oneyear_db()

    Return:
    opencellid_df (pd df):      dataframe to be used in check_online_apis().
    check_online_apis (bool):   indicator for check_online_apis().
    final_df (pd df):           final dataframe is no other checks needed.

    '''
    init_df = init_df_
    init_df = init_df.astype({'mcc': 'Int64', 'mnc': 'Int8', 'lac': 'Int64', 'cid': 'Int64'})
    df = api_cached_oneyear_final_df_

    # Get un-localised cells, get a copy, remove duplicates on cell_id.
    with_missing_df = df[df['location_wgs84.latitude'].isna()].copy()
    with_missing_df = with_missing_df.drop_duplicates(subset=['cell_id'])
    with_missing_df = with_missing_df.astype({'mcc': 'Int64', 'mnc': 'Int8', 'lac': 'Int64', 'cid': 'Int64'})

    # Load openCellID database.
    openCellID = '/home/anon/Desktop/it_stuff/openCellID/cell_towers.parquet'
    ocid_df = pd.read_parquet(openCellID, columns=['mcc', 'net', 'area', 'cell', 'lon', 'lat'])

    with_missing_df = with_missing_df.merge(ocid_df[['mcc', 'net', 'area', 'cell', 'lat', 'lon']],
                                    left_on=['mcc', 'mnc', 'lac', 'cid'],
                                    right_on=['mcc', 'net', 'area', 'cell'],
                                    how='left')

    # Ensure proper handling of NaN values.
    with_missing_df['lat'] = pd.to_numeric(with_missing_df['lat'], errors='coerce')
    with_missing_df['lon'] = pd.to_numeric(with_missing_df['lon'], errors='coerce')

    # Populate coordinates (lat, lon) to empty location_wgs84 when matches occur.
    # Drop nan and duplicates.
    with_missing_df['location_wgs84.latitude'] = with_missing_df['location_wgs84.latitude'].fillna(with_missing_df['lat'])
    with_missing_df['location_wgs84.longitude'] = with_missing_df['location_wgs84.longitude'].fillna(with_missing_df['lon'])
    with_missing_df = with_missing_df.drop(['lat', 'lon', 'cell', 'area', 'net'], axis=1)
    with_missing_df.dropna(subset=['location_wgs84.latitude'], inplace=True)
    with_missing_df.drop_duplicates(subset=['cell_id'], inplace=True)

    # Collect statistical data.
    global opencellid_localised
    stat_df = with_missing_df.drop_duplicates(subset=['cell_id'])
    opencellid_localised = stat_df['location_wgs84.latitude'].isna().value_counts().sum()

    # Get a copy of df from precend stage.
    # Merge new localised coordinates with df copy.
    final_df = df.copy()
    final_df = final_df.merge(with_missing_df[['cell_id', 'location_wgs84.latitude', 'location_wgs84.longitude']],
                        on=['cell_id'], how='left', suffixes=('', '_updated'))

    # Populate coordinates (_updated) to empty location_wgs84 when matches occur, drop _updated.
    final_df['location_wgs84.latitude'] = final_df['location_wgs84.latitude'].fillna(final_df['location_wgs84.latitude_updated'])
    final_df['location_wgs84.longitude'] = final_df['location_wgs84.longitude'].fillna(final_df['location_wgs84.longitude_updated'])
    final_df = final_df.drop(['location_wgs84.latitude_updated', 'location_wgs84.longitude_updated'], axis=1)

    opencellid_df = final_df
    opencellid_df.to_csv('opencellid_df.csv', index=False)


    return opencellid_df


def check_online_apis(api_cached_oneyear_init_df_, opencellid_df_):
    '''
    Check unknown cell-towers against online apis db.

    Parameters:
    api_cached_oneyear_init_df_ (pd df):   data from API_CACHED_ONEYEAR.parquet.
    opencellid_df_ (pd df):           data from openCellID.

    Return:
    api_online_df (pd df):   final dataframe.
    '''

    api_cached_oneyear_init_df = api_cached_oneyear_init_df_
    df = opencellid_df_

    # Get rid off every cell-towers identified by openCellID.
    with_missing_df = df[df['location_wgs84.latitude'].isna()].copy()

    # Create a set with (cell_id, mcc, mnc, lac and cid).
    # Set will get rid off duplicates automatically.
    data = set()
    for _, row in with_missing_df.iterrows():
        cell_data = (row['cell_id'], row['mcc'], row['mnc'], row['lac'], row['cid'])
        data.add(cell_data)

    # HACK: LIMIT DATA TO N NUMBER OF CELLS.
    n = 0
    data = sorted(data)
    data = list(data)[:n]
    db(f"Testing: restricted to {n} cells.\n{data = }", colour='orange_red1')

    if os.path.isfile(API_CACHED_ONEDAY):
        db("check_cached_oneday()")
        in_cached_oneday = check_cached_oneday(data)
        if len(in_cached_oneday) == len(data):
            db("Every cell-tower already checked in the past 24 hours.", colour='orange_red1')
            return df

    else:
        db("check_cached_oneday() not launched")
        # Perform the checks in google and combain apis.
        localisedList, api_localised_df, api_unlocalised_df = check_cell_towers(data)

        # Create dataframe with cell tower locations.
        # Update ifc db with new cell-towers.
        db(f"check_cell_towers(): {len(localisedList) = }")
        cols = ['cell_id', 'lat', 'lon', 'ts', 'source']  # TEST:
        # cols = ['cell_id', 'lat', 'lon', 'ts']
        new_loc_df = pd.DataFrame(localisedList, columns=cols)
        updated_cached_oneyear_df = pd.concat([api_cached_oneyear_init_df, new_loc_df])
        updated_cached_oneyear_df = updated_cached_oneyear_df.sort_values('ts').drop_duplicates(subset=['cell_id'], keep='last')

        # TEST:Collect statistical data.
        global number_cellid
        global n_google
        global google_ratio
        global n_combain
        global combain_ratio

        number_cellid = updated_cached_oneyear_df.shape[0]
        n_google = updated_cached_oneyear_df[updated_cached_oneyear_df['source'] == 'google'].value_counts().sum()
        google_ratio = (n_google * 100) / number_cellid

        n_combain = updated_cached_oneyear_df[updated_cached_oneyear_df['source'] == 'combain'].value_counts().sum()
        combain_ratio = (n_combain * 100) / (number_cellid - n_google)

        db("update API_CACHED_ONEYEAR.parquet")
        updated_cached_oneyear_df.to_parquet(API_CACHED_ONEYEAR, index=False)

        # Create API_CACHED_ONEDAY.parquet only if un-localised cells found.
        # api_unlocalised_df is created in check_cell_towers().
        if api_unlocalised_df.empty:
            db("no un-localised cell-towers found")
            pass
        else:
            db("un-localised cell-towers found, update API_CACHED_ONEDAY.parquet")
            api_unlocalised_df.to_parquet(API_CACHED_ONEDAY, index=False)

    # Merge coordinates found in check_cell_towers()
    with_missing_df = with_missing_df.merge(api_localised_df[['cell_id', 'lat', 'lon']], on=['cell_id'], how='left')

    # with_missing_df.to_csv('withmissing2.csv', index=False)
    # # Ensure proper handling of NaN values.
    with_missing_df['lat'] = pd.to_numeric(with_missing_df['lat'], errors='coerce')
    with_missing_df['lon'] = pd.to_numeric(with_missing_df['lon'], errors='coerce')

    # Populate coordinates (lat, lon) to empty location_wgs84 when matches occur.
    # Drop un-necessary columns.
    with_missing_df['location_wgs84.latitude'] = with_missing_df['location_wgs84.latitude'].fillna(with_missing_df['lat'])
    with_missing_df['location_wgs84.longitude'] = with_missing_df['location_wgs84.longitude'].fillna(with_missing_df['lon'])
    with_missing_df = with_missing_df.drop(['lat', 'lon'], axis=1)
    # with_missing_df.to_csv('withmissing3.csv', index=False)

    with_missing_df.dropna(subset=['location_wgs84.latitude'], inplace=True)
    with_missing_df.drop_duplicates(subset=['cell_id'], inplace=True)
    with_missing_df.to_csv('withmissing4.csv', index=False)

    final_df = df.copy()
    final_df = final_df.merge(with_missing_df[['cell_id', 'location_wgs84.latitude', 'location_wgs84.longitude']],
                        on=['cell_id'], how='left', suffixes=('', '_updated'))
    final_df.to_csv('final_tmp_df.csv', index=False)

    final_df['location_wgs84.latitude'] = final_df['location_wgs84.latitude'].fillna(final_df['location_wgs84.latitude_updated'])
    final_df['location_wgs84.longitude'] = final_df['location_wgs84.longitude'].fillna(final_df['location_wgs84.longitude_updated'])
    api_online_df = final_df.drop(['location_wgs84.latitude_updated', 'location_wgs84.longitude_updated'], axis=1)

    api_online_df.to_csv('api_online_df.csv', index=False)

    return api_online_df


def check_cached_oneday(data_):
    '''
    Check non-localised cells against 24cellt.parquet.

    Parameters:
    data_ (set):    un-localised data.

    Return:
    in_cached_oneday_set (set): cell-towers found in API_CACHED_ONEDAY.parquet.

    '''
    # Load 24cellt.parquet data.
    api_cached_oneday_df = pd.read_parquet(API_CACHED_ONEDAY)
    ts_cut = (int(time.time()) - 86400) # 1 day.
    filt = (api_cached_oneday_df['ts'] > ts_cut)
    df = api_cached_oneday_df[filt]

    in_cached_oneday_set = set()
    cellt_list = list(data_)
    for cell in cellt_list:
        db(cell[0])
        if cell[0] in df['cell_id'].values:
            in_cached_oneday_set.add(cell)

    # Load current API_CACHED_ONEDAY.parquet and update.
    cols = ['cell_id', 'ts']
    now_in_cached_oneday_df = pd.DataFrame(in_cached_oneday_set, columns=cols)
    now_in_cached_oneday_df['ts'] = int(time.time())
    updated_cached_oneday_df = pd.concat([api_cached_oneday_df, now_in_cached_oneday_df])
    updated_cached_oneday_df = updated_cached_oneday_df.sort_values('ts').drop_duplicates(subset=['cell_id'], keep='last')
    updated_cached_oneday_df.to_parquet(API_CACHED_ONEDAY, index=False)

    return in_cached_oneday_set


def check_cell_towers(cellTower_dataList_):
    '''
    Take cell tower data list from split_process_concat_dataframe().
    Determine what checks needs performing (ifc_db or apis).
    Feed api_requester() with a list of cell-towers, parse answers.

    Parameters:
    cellTower_dataList_ (list), list of cell-towers to be checked.

    Return:
    localised (list): data format of cell-towers (cell_id, lat, lon, ts).
    api_localised_df (pd df):   dataframe of localised cell-towers by apis.
    api_unlocalised_df (pd df): dataframe of un_localised cell-towers.
    '''

    ctdl = cellTower_dataList_

    global launch_google_api
    global launch_combain_api
    global error_google_api
    global error_combain_api

    localised = []
    not_localised = []

    # Determine if google and combain works properly.
    # Do not put inside while loop.
    # False: errors 400 or 403.
    error_google_api = False
    error_combain_api = False
    i = 0
    while i < len(ctdl):
        launch_google_api = True        # launch_google_api = False
        launch_combain_api = False

        cellTowerData = [
            {
                "mobileCountryCode": list(ctdl)[i][1], # mcc.
                "mobileNetworkCode": list(ctdl)[i][2], # mnc.
                "locationAreaCode": list(ctdl)[i][3],  # lac.
                "cellId": list(ctdl)[i][4]             # cid.
            }
        ]

        # db(f"mcc: {list(ctdl)[i][1]}\nmnc: {list(ctdl)[i][2]}\nlac: {list(ctdl)[i][3]}\ncid: {list(ctdl)[i][4]}")

        # Google Api.
        if launch_google_api and not error_google_api:
            db("Google API", colour='green')
            GOOGLE_API_KEY = constants.GOOGLE_API_KEY
            url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
            result = api_requester('google', url, cellTowerData)
            if result:
                lat = result['location']['lat']
                lon = result['location']['lng']
                ts = int(time.time())
                localised.append([list(ctdl)[i][0], lat, lon, ts, 'google']) # TEST:

        # Combain api.
        if launch_combain_api and not error_combain_api:
        # if not error_combain_api:
            db("Combain API", colour='orange_red1')
            COMBAIN_API_KEY = constants.COMBAIN_API_KEY
            url = f"https://apiv2.combain.com?key={COMBAIN_API_KEY}"
            result = api_requester('combain', url, cellTowerData)
            if result:
                lat = result['location']['lat']
                lon = result['location']['lng']
                ts = int(time.time())
                localised.append([list(ctdl)[i][0], lat, lon, ts, 'combain']) # TEST:
            else:
                not_localised.append(list(ctdl)[i][0])

        # Do not make api requests anymore.
        if error_google_api and error_combain_api:
            break

        i += 1

    # Create dataframe with cell tower locations.
    # cols = ['cell_id', 'lat', 'lon', 'ts']
    # api_localised_df = pd.DataFrame(localised, columns=cols)
    # TEST:
    cols = ['cell_id', 'lat', 'lon', 'ts', 'source']
    api_localised_df = pd.DataFrame(localised, columns=cols)

    # Create dataframe with un-localised cell_id.
    cols = ['cell_id']
    api_unlocalised_df = pd.DataFrame(not_localised, columns=cols)
    api_unlocalised_df['ts'] = int(time.time())

    return localised, api_localised_df, api_unlocalised_df


def api_requester(api, url_, cellTower_data):
    '''
    Handle POST requests process on Cell-Towers db and apis.

    Called in check_cell_towers().

    Parameters:
    api: str, name of api being checked
    cellTower_data: dict, contains unique cell tower data
    max_retries: int.
    '''

    global launch_google_api
    global launch_combain_api
    global error_google_api
    global error_combain_api

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
            errorMsg = dedent('''\
                                400 = Parse Error / Invalid key.
                                403 = Out of credits.''')

            if status_code in (400, 403):
                if api == 'google':
                    db(f"Something went wrong with Google api: {status_code = }\n{errorMsg}", colour='red')
                    error_google_api = True
                    launch_combain_api = True
                    # break
                elif api == 'combain':
                    db(f"Something went wrong with Combain api: {status_code = }\n{errorMsg}", colour='red')
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
    df.to_csv('s2_cell_data.csv', index=False)

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
    celldf.to_csv('s3_celldf.csv', index=False)

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


def summary():
    '''Return some statistics.'''

    # INFO:
    # To get accurate ratios for apis, I must also include the number of non-localised
    # cells from API_CACHED_ONEDAY.parquet.
    # TODO: Verify if the above statement is correct.

    def ratios(n_by_api):
        '''Return ratios.'''
        ratio = (n_by_api * 100) / missing_cells
        return f"{ratio:.2f}%"

    # Number of unique un-localised cell-towers.
    print(f"Unique un-localised cell-towers: {missing_cells}")
    print(f"Cell-towers identified by openCellId: {opencellid_localised} {ratios(opencellid_localised)}")
    print(f"Cell-towers identified by Google: {n_google} {ratios(n_google)}")
    print(f"Cell-towers identified by Combain: {n_combain} {ratios(n_combain)}")


def mcc_checker(initdf_, finaldf_, cell_counter_dic):
    '''Statistics on cell-towers and localised ratios.'''

    # Counters for initial cells do not change over time.
    # Values comes from json_to_dataframe().
    totCell = cell_counter_dic

    # Get the list of unique mcc.
    findf = finaldf_
    findf['mcc'] = findf['mcc'].astype(str)
    mcc_list = findf['mcc'].unique()

    # Create the data structure for statistics.
    data = []
    for mcc in mcc_list:
        filt = (findf['mcc'] == mcc)
        country_name = mobile_codes.mcc(mcc)[0].name if mobile_codes.mcc(mcc) else "UNKNOWN"
        unique_cells = findf[filt]['cell_id'].nunique()
        # unique_cell_df = findf[findf['mcc'] == int(mcc)].drop_duplicates(subset=['cell_id'])
        unique_cell_df = findf[filt].drop_duplicates(subset=['cell_id'])
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
    # with console.status("[bold][italic][green]Processing iri file...[/]"):
    # unzip_file(zipFile)
    # find_iri_csv()

    csv_to_json(iriFile, iriJsonFile)
    initial_df, counterDic = json_to_dataframe(iriJsonFile)

    api_cached_oneyear_init_df, api_cached_oneyear_final_df = check_cached_oneyear_db(initial_df)
    opencellid_df = check_opencellid(initial_df, api_cached_oneyear_final_df)
    api_online_df = check_online_apis(api_cached_oneyear_init_df, opencellid_df)

    transpose_cells_on_map(api_online_df)

    summary()

    mcc_checker(initial_df, api_online_df, counterDic)

# TODO: add an option to erase non-necessary files.
# Modify argMessage.
if __name__ == "__main__":
    argMessage = dedent('''\
                        Script that format iri.csv to json then parses the data to retrieve IMEI.''')
    parser = ArgumentParser(description=argMessage, formatter_class=RawTextHelpFormatter)
    parser.usage = ("Takes path to iri file as argument")
    args = parser.parse_args()

    main()

    # rem(){ rm *cell* *df* with* 2> /dev/null;}
    # setup(){ py iris.py; }
