from collections import UserDict
class Incar(UserDict):
   
    def __init__(self,name='',dict=None,finish=False,path=None):
        UserDict.__init__(self,dict)
        self.__name = name
        self.__path = path
        self._group = {}
        self.finish = finish

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return self.__name

    @property
    def group(self):
        return self._group

    def downdate(self,params):
        from copy import deepcopy

        if isinstance(params, Incar):
            params = deepcopy(params.data)

        cache = self.data
        self.data = params
        self.data.update(cache) 

    def update_params(self, params):

        if isinstance(params, Incar) and params.name != '':
            self.data.update(params)
            self.group[params.name] = list(params.keys())


class Task(UserDict):

    def __init__(self,soft=None,dict=None):
        UserDict.__init__(self,dict)
        self.__soft = soft
        self._builder = None

    @property
    def finish(self):
        fin = True
        for key in self.data:
            if self.data[key].finish != True:
                fin = False
                break
        return fin

    @finish.setter
    def finish(self,value):
        assert isinstance(value,bool)
        for key in self.data:
            self.data[key].finish = value

    @property
    def soft(self):
        return self.__soft

    @property
    def builder(self):
        return self._builder

    def __getattr__(self,task):

        if task in self.data:
            return self.data[task]
        else:
            # raise AttributeError("Invaild property %s." %task)
            return None

    def __deepcopy__(self,memo):
        from copy import deepcopy
        cp = {}
        for key, value in self.data.items():
            cp[key] = deepcopy(value)
        return Task(self.soft,cp)

    def set_energy(self,energy):    
        self.builder.set_energy(self,energy)

    def set_force(self,force):    
        self.builder.set_force(self,force)
