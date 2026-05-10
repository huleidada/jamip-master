# -*- coding: utf-8 -*-
#!/usr/bin/env python3

class SubmitScript:
    def __init__(self, path, filename='submit.sh', **kwargs):
        """
        submit script for task.
        
        Arguments:
            path: the path of outputing.
            filename (default='submit.sh'): filename of submit script file.
        """
        self.path=path
        self.filename=filename
        
        if 'type' in kwargs:
            if kwargs['type'].lower() == 'pbs':
                self.set_submit('pbs')
                
    def set_submitScript(self, parameters):
        """
        set parameters of submit script.
        
        Arguments:
            parameters: 
        """
        pass
    
    def output(self):
        """
        output Submit file.
            
        Return:
            Submit's object.
        """
        path=self.path
        filename=self.filename
        
        with open('{}/{}'.format(path, filename), 'w') as f:
            f.write('')
        return self
