from SendMail import SendMail
from ReceiveMail import ReceiveMail
from pymongo import MongoClient
from User.User import PersonRasp
import threading
import time
import datetime as dt
import hashlib
import io
import pickle
import mongoengine as me
import numpy
import matplotlib.pyplot as plt


## Connect to DBs
# Connection to mongoengine DB Rasp
mongo_client_Rasp = MongoClient('mongodb://grupo14.duckdns.org:1226')
db_Rasp = mongo_client_Rasp["Test1"]
col_Rasp = db_Rasp["person"]
print(f"[INFO] Conecting to DB Rasp with:{col_Rasp.read_preference}...")

mongo_client = MongoClient('mongodb://grupo14.duckdns.org:1226')
db = mongo_client["Test1"]


idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
PersonRasp(idrasp= idh,last_in = dt.datetime.utcnow, is_recognized=False, likelihood=0.85).save()
