import sqlite3
from pprint import pprint

def GetTables(db_file = 'db/jamipdb'):
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("select name from sqlite_master where type='table' order by name")
        pprint(cur.fetchall())
    except sqlite3.Error as e:
            print(e)

def GetTable_info(db_file = 'db/jamipdb',table = 'structure'):
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("select * from sqlite_master where name = '%s'" %table)
        pprint(cur.fetchall())
    except sqlite3.Error as e:
            print(e)

def GetAlldata(db_file = 'db/jamipdb',table = 'structure'):
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("select * from '%s'" %table)
        pprint(cur.fetchall())
    except sqlite3.Error as e:
            print(e)

GetTables()
#GetTable_info()
GetTable_info(table='entry')
GetAlldata(table='entry')
