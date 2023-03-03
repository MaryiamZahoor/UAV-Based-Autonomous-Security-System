import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('fypapi-275115-1487f536b9fc.json', scope)

gc = gspread.authorize(credentials)

wks = gc.open("droneCordinates").sheet1

#print(wks.get_all_records())

#values_list = wks.row_values(2)

#if values_list:
#    print(values_list)

x=2515
y=1254

wks.append_row([x,y])

#wks.delete_row(2)

# pip install PyOpenSSL
# pip install oauth2client
# pip install gspread

'''x = wks.cell(5,1).value
y=20

if x and y:
    x = float(x)   
    y=y+x
    print(y)'''