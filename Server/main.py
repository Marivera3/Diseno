from SendMail import SendMail
from ReceiveMail import ReceiveMail
from pymongo import MongoClient
from User.User import PersonRasp, PersonServ, RegisteredUser
import threading
import time
import datetime as dt
import hashlib
import io
import pickle
import mongoengine as me
import numpy
import matplotlib.pyplot as plt





user = 'disenouc20g14@gmail.com'
# password = getpass.getpass()
password = 'DisenoUCg14'

# admin
receiver_email = 'max.rivera.figueroa@gmail.com'

# Connection to mongoengine
me.connect('test1')
# Connection to pymongo
mongo_client = MongoClient('mongodb://localhost:27017')
db = mongo_client["test1"]
col = db["person_rasp"]

# Start Rx-threading
receive = ReceiveMail(user=user, password=password, time=1)
receive.start()

def readrxbuffer(rx, col):
    if len(rx.list_msg) > 0:
        for i in rx.list_msg:
            print(f'RX = Name: {i.name}, Surname: {i.surname}, BD: {i.bd}, ID: {i.id}')
            if i.bd == 1:
                # Significa que la persona no reconocida ya esta agregada a la
                # base de datos. Se debe agregar la serialize_pic de i.id con
                # la bd
                pass


            myquery = {"idrasp": i.id, "waiting_response": True}
            newvalues = {"$set": {
                                 "name": i.name,
                                 "surname": i.surname,
                                 "mail_recieved": True,
                                 "mail_sent": True,
                                 "waiting_response": False
                                 }}
            x = col.update_one(myquery, newvalues)
            print(x.modified_count, "documents updated.")
        rx.clear_msg()



def addperson2db(col, emb, rgb, rec=False, lh=0):
    # name is
    idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
    PersonRasp(idrasp= idh,last_in = dt.datetime.utcnow, is_recognized=rec, seralize_pic=emb, picture=str(rgb), likelihood=lh).save()
    if rec == False:
        mail1 = SendMail(user=user, password=password,
                        receiver=receiver_email, id=idh, col=col)
        mail1.setmessage('content.txt')
        buffer = io.BytesIO()
        plt.imsave(buffer, rgb)
        mail1.attach_bytes(buffer.getbuffer(), 'unknown.png')
        mail1.start()
    return idh

data = pickle.loads(open('imagen.pickle', "rb").read())
# addperson2db(col, emb = '', rgb=data, rec=False, lh=0.0)
#
c = 0
while True:
    c += 1
    time.sleep(1)
    readrxbuffer(receive, col)
    if c == 300:
        receive.disconnectserver()
        break
