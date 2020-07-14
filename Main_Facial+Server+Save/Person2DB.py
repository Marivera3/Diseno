import threading
import codecs
import hashlib
import io
import pickle
import queue
import time
import datetime as dt
import mongoengine as me
from pymongo import MongoClient
from User.User import PersonRasp

class Person2DB(threading.Thread):

        def __init__(self, name, queue_pframes):
            # '-t', str(time)
            super().__init__(name=name)
            self.queue_pframes = queue_pframes
            # self.host = 'mongodb://grupo14.duckdns.org:1226/Rasp'
            self.host = '192.168.0.242'
            self.port = 27017
            self.db = 'test1'
            self.stopped = False



        def addperson2db(self, name, surname, is_recongized, last_in, picture, likelihood):
            idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
            # only entries, via mongoengine
            if is_recongized:
                PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                            is_recognized=is_recongized, likelihood=likelihood).save()

            else:
                PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                            is_recognized=is_recongized, likelihood=likelihood,
                            picture=picture).save()

        def modifypersondb(name, surname, idh, last_out, likelihood):
            # rutine with pymongo
            pass


        def run(self):
            c = 0
            while True:
                connected = True
                try:
                    elem = self.queue_pframes.get(False)
                    c = 0
                    for item in elem:
                        while connected:
                            try:
                                print('[INFO] Connecting to DB mongoengine...')
                                db_client = me.connect(self.db, host=self.host , port=self.port)
                                # db_client = me.connect('test1', host='192.168.0.242', port=27017)
                                # time.sleep(0.05)
                                connected = False

                                # Falta rutina para ver si es unkown o no
                                # Falta rutina para identificar solo 1 unknown (o solo 1 persona una vez y no multiples veces)
                                # Falta rutina para salida

                                # pickled = codecs.encode(pickle.dumps(item[0]), "base64").decode()
                                print(f'nombre: {item[3]}, Likelihood: {item[4]}')
                                self.addperson2db(name=item[3], surname='', is_recongized=False,
                                            last_in=item[5],
                                            picture="", likelihood=item[4])
                                # time.sleep(0.05)
                                print('[INFO] Connection succed')
                                db_client.close()

                            except:
                            # Falta fijar la excepcion
                                print('[INFO] Connection fails')
                                connected = True
                except queue.Empty:
                    time.sleep(1)
                    c += 1
                    print(f'c_db : {c}')
                    if c > 10:
                        print('break cicle')
                        break
