from User.User import PersonRasp
import datetime as dt
import hashlib
import io
import time
import pickle
import mongoengine as me
import numpy


## Connect to DBs
# Connection to mongoengine DB Rasp
# me.connect('Rasp', host='grupo14.duckdns.org', port=1226)
me.connect('Rasp')
idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
PersonRasp(idrasp= idh,last_in = dt.datetime.utcnow, is_recognized=False, likelihood=0.85, name='Max', surname='Rivera').save()
