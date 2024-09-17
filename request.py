import os
from os.path import isfile
import requests
import sys
import time
import pandas as pd
from rich import print as rprint
from mydebug import db

# Both Google and Combain apis share the same error codes.
# 400 = Parse Error / Invalid key.
# 403 = Out of credits.
# 404 = Not found (meaning cell tower not found, api is ok).

# Exit codes:
# 0     normal exit.
# 1     Internet connection issues.
# 2     api issues (limit or key).

IFC_CELL_DB = '/home/anon/Documents/git/pythonScripts/iris/IFC_CELLT_DB.csv'

# Counters for statistics.
ifc3_localised = 0
ifc3_checked = 0
google_localised = 0
google_checked = 0
combain_localised = 0
combain_checked = 0

# 222 is fake (warning: found by combain though!).
# cellTower_dataList = set([('230-01-107825008', 230, 1, 14418, 107825008),
#             ('230-01-14418-21332', 230, 1, 14418, 21332),
#             ])
# ('222-1-9031-1280525', 222, 1, 9031, 1280525)

cellTower_dataList = set([('230-01-107825008', 230, 1, 14418, 107825008),
            ('230-01-14418-21332', 230, 1, 14418, 21332),
            ('222-1-9031-1280525', 222, 1, 9031, 1280525)
            ])


def api_requester(api, url_, cellTower_data):
    '''
    Handle POST requests process on Cell Towers tools.

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
    Feed api_requester() with a list of cell towers, parse answers.

    cellTower_dataList: list, list of cell towers.

    Return:
    localised: list, data format of cell towers.
    ifc_df: pandas df, newly identified cell towers.
    '''

    ctdl = cellTower_dataList_

    # ctFile = 'cs_cell_towers.csv'
    global ifc3_localised
    global google_localised
    global combain_localised
    global ifc3_checked
    global google_checked
    global combain_checked
    global launch_google_api
    global launch_combain_api

    # Load ifc3 own database.
    if os.path.isfile(IFC_CELL_DB):
        db(f"{IFC_CELL_DB} does exist.", colour='green')
        ifc_df = pd.read_csv(IFC_CELL_DB)
        # Filter on current year.
        # ts_cut = int(time.time()) - 20 # test: get rid of files older than 60 seconds. 
        ts_cut = int(time.time()) - 31536000 # 1 year.
        filt = (ifc_df['ts'] > ts_cut)
        ifc_df = ifc_df[filt]
        ifc_df.astype({'cell_id': str, 'lat': 'Float64', 'lon': 'Float64', 'ts': 'Int64'})

    # Database does not exist, create empty template with columns.
    else:
        db(f"{IFC_CELL_DB} does not exist.", colour='red')
        cols = ['cell_id', 'lat', 'lon', 'ts']
        ifc_df = pd.DataFrame(columns=cols)

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
            db(f"{cell_id} in ifc_df", colour='green')
            filt = (ifc_df['cell_id'] == cell_id)
            lat = (ifc_df[filt]['lat'].values)[-1]
            lon = (ifc_df[filt]['lon'].values)[-1]
            ts = (ifc_df[filt]['ts'].values)[-1]
            # db(f"{cell_id = }\n{lat = }\n{lon = }\n{ts = }")
            # db(f"{type(cell_id)}\n{type(lat)}\n{type(lon)}\n{type(ts)}")
            localised.append([cell_id, lat, lon, ts])
            ifc3_localised += 1
            ifc3_checked += 1
        else:
            db(f"{cell_id} not in ifc_df", colour='red')
            launch_google_api = True

        # Google Api.
        if launch_google_api and not error_google_api:
            GOOGLE_API_KEY = "ADD_API_KEY_HERE"
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
            COMBAIN_API_KEY = "ADD_API_KEY_HERE"
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

    return localised, ifc_df


def create_dataframe(locdata, ifcdf):
    '''Create dataframe with cell tower locations.'''
    cols = ['cell_id', 'lat', 'lon', 'ts']
    df = pd.DataFrame(locdata, columns=cols)
    new_ifc_df = df.combine_first(ifcdf)
    new_ifc_df.to_csv(IFC_CELL_DB, index=False)

    return df


def summary():
    '''Return statistics.'''

    def api_stat(api, localised, checked):
        '''Return counts and percentage per api'''
        percentage = 0
        if checked != 0:
            percentage = (localised * 100) / checked
            print(f"Number of cell towers identified by {api}: {localised}/{checked} ({percentage:.2f}%)")
        else:
            print(f"No cell towers checked with {api}.")

    print(f"Total cell towers checked: {len(cellTower_dataList)}")
    api_stat('IFC_CELL_DB', ifc3_localised, ifc3_checked)
    api_stat('Google', google_localised, google_checked)
    api_stat('Combain', combain_localised, combain_checked)

    # This is redundant of api_requester().
    if launch_google_api:
        if google_checked != len(cellTower_dataList):
            rprint(f"Inconsistent results!")

    return None


def main():
    '''Script launcher.'''
    loc_data, ifc_df = check_cell_towers(cellTower_dataList)
    create_dataframe(loc_data, ifc_df)
    print()
    summary()


if __name__ == "__main__":
    main()
    sys.exit(0)
