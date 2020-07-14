import threading
import requests
import json
import time
import datetime


class CheckDB2(threading.Thread):

        def __init__(self, name, seconds=2):
            super().__init__(name=name)
            self.__url_DB     = "http://server4diseno.duckdns.org:1227/DB"
            self.__url_Number = "http://server4diseno.duckdns.org:1227/Number"
            self.stopped    = False
            self.actual_dbsize = 0
            self.prev_time = datetime.datetime.utcnow()
            self.has_changes = False
            self.db = None
            self.seconds = seconds

        def run(self):
            while True:
                if (datetime.datetime.utcnow() - self.prev_time).total_seconds() > self.seconds:
                    self.prev_time = datetime.datetime.utcnow()
                    try:
                        myfile = requests.get(self.__url_Number)
                        if myfile.status_code == 200:
                            new_size = int(myfile.content.decode())
                            if new_size > self.actual_dbsize:
                                database = requests.get(self.__url_DB)
                                if database.status_code == 200:
                                    self.actual_dbsize = int(myfile.content.decode())
                                    self.has_changes = True
                                    self.db = json.loads(database.content.decode())
                            else:
                                self.has_changes = False
                    except:
                        print('[INFO] Could no connect to Server Database ')
                    time.sleep(1)
