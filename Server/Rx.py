class Rxmsg:


    def __init__(self, name=None, surname=None, BD=0, ID=None):

        self.name = name
        self.surname = surname
        self.bd = BD
        self.id = ID

    def readsubject(self, subj):
        self.id = subj.strip().split('-')[-1].strip().split(';')[1].split()[0]

    def readbody(self, body):
        rx = body.strip().split(';')[0].strip('[').strip(']').split(',')
        for att in rx:
            attaux = att.strip().split(':')
            if attaux[0] == 'Nombre':
                self.name = attaux[1].split()[0]
            elif attaux[0] == 'Apellido':
                self.surname = attaux[1].split()[0]
            elif attaux[0] == 'BD':
                self.bd = attaux[1].split()[0]
            elif attaux[0] == 'ID':
                self.id = attaux[1].split()[0]
