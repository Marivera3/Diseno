import threading
import smtplib
import ssl
import getpass
import email
from pymongo import MongoClient
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class SendMail(threading.Thread):


    def __init__(self, user, password, receiver, id, col):
        super().__init__()
        self.user           = user
        self.__pass         = password
        self.receiver_email = receiver
        self.id             = id
        self.subject        = ''
        self.body           = ''
        self.filename       = ''
        self.message        = None
        self.smtp_server    = "smtp.gmail.com"
        self.port           = 465
        self.collection     = col

    def setcontent(self, file):
        with open("content.txt", "r", encoding='utf-8') as f:
            self.body= f.read()

    def setmessage(self, body):
        self.setcontent(body)
        self.subject = "Oficina-1: PNID"
        self.message = MIMEMultipart()
        self.message["Subject"] = self.subject
        self.message["From"] = self.user
        self.message["To"] = self.receiver_email
        self.message.attach(MIMEText(self.body.format(id=self.id), "plain"))



    def attach_file(self, file):
        # If file are already in bytes, use bool=1, if yoou want to read the
        # file from an extendion, i.e, jpeg, use bool=0
        # filename is the string name of the file with the extension
        with open(file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {file}",
            )
            self.message.attach(part)

    def attach_bytes(self, file, filename):
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file)

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )
        self.message.attach(part)

    def connect(self):
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
            server.login(self.user, self.__pass)
            server.sendmail(self.user, self.receiver_email, self.message.as_string())
            if server.noop()[0] == 250:
                x = self.setmailsent()
                print(f'Mail sent...')
                print(x.modified_count, "documents updated.")

    def run(self):

        self.connect()


    def setmailsent(self):
            myquery = {"idrasp": self.id}
            newvalues = {"$set": {
                                 "mail_sent": True,
                                 "waiting_response": True
                                 }}
            x = self.collection.update_one(myquery, newvalues)
            return x
