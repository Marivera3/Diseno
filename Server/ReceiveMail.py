import threading
import imaplib
import email
from email.header import decode_header
import webbrowser
import json as js
import Rx
import time


class ReceiveMail(threading.Thread):


    def __init__(self, user, password, time=5):
        super().__init__()
        self.user                   = user
        self.__pass                 = password
        self.time                   = time
        self.imap_server            = "imap.gmail.com"
        self.total_receive_mess     = 0
        self.old_total_receive_mess = 0
        self.new_message            = 0
        self.count                  = 0
        self.disconnect             = False
        self.list_msg               = []

    def run(self):
        self.connect()

    def disconnectserver(self):
        self.disconnect = True

    def clear_msg(self):
        self.list_msg = []

    def connect(self):
        self.disconnect = False
        self.old_total_receive_mess = 0

        with imaplib.IMAP4_SSL(self.imap_server) as imap:
            imap.login(self.user, self.__pass)
            while not self.disconnect:
                time.sleep(self.time)
                status, messages = imap.select("INBOX")
                self.total_receive_mess = int(messages[0])

                if self.old_total_receive_mess == self.total_receive_mess:
                    self.new_message = self.total_receive_mess - self.old_total_receive_mess
                    print(f'No new messages')

                elif self.old_total_receive_mess < self.total_receive_mess:
                    self.new_message = self.total_receive_mess - self.old_total_receive_mess
                    print(f'We have new mess: {self.new_message}')

                # print(f'Total Receive Message: {self.total_receive_mess}')
                self.old_total_receive_mess = self.total_receive_mess

                for i in range(self.total_receive_mess, self.total_receive_mess-self.new_message, -1):
                    if i<=0:
                        continue

                    res, msg = imap.fetch(str(i), "(RFC822)")

                    self.list_msg.append(Rx.Rxmsg())

                    for response in msg:
                        if isinstance(response, tuple):
                            # parse a bytes email into a message object
                            msg = email.message_from_bytes(response[1])
                            # subject = decode_header(msg["Subject"])[0][0]
                            '''if isinstance(subject, bytes):
                                # if it's a bytes, decode to str
                                print('subject in bytes')
                                subject = subject.decode()'''
                            # from_ = msg.get("From")
                            # self.list_msg[-1].readsubject(subject)

                            if msg.is_multipart():

                                for part in msg.walk():
                                    # extract content type of email
                                    # print(f'part: {part}') # print everything of the received mail
                                    content_type = part.get_content_type()
                                    content_disposition = str(part.get("Content-Disposition"))
                                    try:
                                        # get the email body
                                        body = part.get_payload(decode=True).decode()
                                    except:
                                        pass

                                    if content_type == "text/plain" and "attachment" not in content_disposition:
                                        # print text/plain emails and skip attachments

                                        self.list_msg[-1].readbody(body)
