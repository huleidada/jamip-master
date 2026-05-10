import os
import shutil

class __CreateInput(object):

    '''
    copy scripts to Current Directory
    '''

    def __init__(self,parse):

        self.output = os.getcwd()
        if parse['input'] == 'vasp':
            self.copyfile = ['vasp.py']
            self.output += '/input.py'

        elif parse['input'] == 'qe':
            self.copyfile = ['qe.py']
            self.output += '/input.py'

        elif parse['input'] == 'cp2k':
            self.copyfile = ['cp2k.py']
            self.output += '/input.py'

        elif parse['input'] == 'gaussian':
            self.copyfile = ['gaussian.py']
            self.output += '/input.py'

        elif parse['input'] == 'plot':
            self.copyfile = ['plot.py']

        # elif parse['input'] == 'db':
        #     self.copyfile = ['dbshell.py','dbload.py']

        else:
            self.copyfile = []

        self.CopyFiles()

    def CopyFiles(self):
        home = os.environ['HOME']+'/.jamip/utils/'
        for file in self.copyfile:
            shutil.copy(home+file,self.output)
