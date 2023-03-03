import airsim

import sys
import time
import math

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from goto import with_goto


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('fypapi-275115-1487f536b9fc.json', scope)

gc = gspread.authorize(credentials)

wks = gc.open("droneCordinates").sheet1


''' 0.00520833335 '''


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    client.hoverAsync().join()


def track_person(y,x):
    '''client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)'''
    start = client.getMultirotorState().kinematics_estimated.position
    z = -5
    start.x_val = start.x_val + ((-x) * 0.0104166667)
    start.y_val = start.y_val + (y * 0.0104166667) 
    client.moveToPositionAsync(start.x_val,start.y_val,z,1).join()

@with_goto
def startFunc():
    label .start_the_tracking

    x = wks.cell(2,1).value
    y = wks.cell(2,2).value
    '''for x,y in arr:
        track_person(y,x)'''

    if x and y:
        x = float(x)
        y = float(y)
        track_person(x,y)

        wks.delete_row(2)

    goto .start_the_tracking

startFunc()

'''
if(var == True):
    track_person(x,y)
else:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    var=True
    track_person(x,y)'''