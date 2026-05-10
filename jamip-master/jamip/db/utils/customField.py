import numpy as np
import json
import pickle, base64
from django.db import models

class ListField(models.TextField):
    description='Stores a python list'
 
    def __init__(self, *args, **kwargs):
        super(ListField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return []
        
        return pickle.loads(base64.b64decode(value))
    
    def to_python(self, value):
        if value is None:
            return []
 
        if isinstance(value, list):
            return value
 
        try:
            return pickle.loads(str(value))
        except:
            return json.loads(value)
    
    def get_prep_value(self, value):
        data=None
        if value is None:
            pass
        elif isinstance(value, list):
            data=pickle.dumps(value)
        elif isinstance(value, np.ndarray):
            data=pickle.dumps(value.tolist())
        else:
            raise TypeError('%s is not a list or numpy array' %value)
        return data if data is None else base64.b64encode(data)
 
    #def value_to_string(self, obj):
    #    value=self._get_val_from_obj(obj)
    #    return self.get_db_prep_value(value)
        
class NumpyArrayField(models.TextField):
    description='Stores a numpy ndarray'

    def __init__(self, *args, **kwargs):
        super(NumpyArrayField, self).__init__(*args, **kwargs)
    
    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return np.array([])   
# =============================================================================
#         return np.array(pickle.loads(str(value)))
# =============================================================================
        return np.array(pickle.loads(base64.b64decode(value)))
    
    def to_python(self, value):
        import re
        if value is None or value == '[]':
            return np.array([])
            
        if isinstance(value, list):
            value=np.array(value)
        if isinstance(value, np.ndarray):
            return value
        
        try:
            return np.array(pickle.loads(str(value)))
        except:
            value = re.sub(r'\.?\s+', ",", value)
            value = re.sub(r',]', "]", value)
            value = re.sub(r'\[,', "[", value)
            return np.array(json.loads(value))
    
    def get_prep_value(self, value):
# =============================================================================
#         if value is None:
#             return value
#         elif isinstance(value, list):
#             return pickle.dumps(value, protocol=0)
#         elif isinstance(value, np.ndarray):
#             return pickle.dumps(value.tolist(), protocol=0)
#         else:
#             raise TypeError('%s is not a list or numpy array' %value)
# =============================================================================
        data=None
        if value is None:
            pass
        elif isinstance(value, list):
            data=pickle.dumps(value)
        elif isinstance(value, np.ndarray):
            data=pickle.dumps(value.tolist())
        else:
            raise TypeError('%s is not a list or numpy array' %value)
        return data if data is None else base64.b64encode(data)
    
    #def value_to_string(self, obj):
    #    value = self.value_from_object(obj)
    #    return self.get_prep_value(value)
        
class DictField(models.TextField):
    description='Stores a python dictionary'
    
    def __init__(self, *args, **kwargs):
        super(DictField, self).__init__(*args, **kwargs)
    
    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return {}
        if isinstance(value, str):
            return pickle.loads(str(value))
        else:
            return pickle.loads(value)
    
    def to_python(self, value):
        if value is None or value == '{}':
            return {}
            
        if isinstance(value, dict):
            return value
        
        try:
            return pickle.loads(value)
        except:
            value = value.replace('\'','\"')
            return json.loads(value)
    
    def get_prep_value(self, value):
        if value is None:
            return value
        elif isinstance(value, dict):
            return pickle.dumps(value, protocol=0)
        else:
            raise TypeError('%s is not a python dictionary' %value)
    
    #def value_to_string(self, obj):
    #    value=self._get_val_from_obj(obj)
    #    return self.get_db_prep_value(value)
    
class CompressedTextField(models.TextField):
    description='Stores a string text in a compressed format (bz2 by default)'
    
    def __init__(self, *args, **kwargs):
        super(CompressedTextField, self).__init__(*args, **kwargs)
        
    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return ''
        
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def to_python(self, value):
        if value is None:
            return ''
        
        if isinstance(value, str):
            return value
        
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def get_prep_value(self, value):
        if value is None:
            return value
        elif isinstance(value, str):
            return pickle.dumps(value, protocol=1).encode('utf-8').encode('bz2').encode('base64')
        else:
            raise TypeError('%s is not a string text' %value)

class CompressedListField(models.TextField):
    description='Stores a python list in a compressed format (bz2 by default)'
 
    def __init__(self, *args, **kwargs):
        super(CompressedListField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return []
        
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def to_python(self, value):
        if value is None:
            return []
 
        if isinstance(value, list):
            return value
 
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def get_prep_value(self, value):
        if value is None:
            return value
        elif isinstance(value, list):
            #string=pickle.dumps(value, protocol=1)
            #return string.encode('utf-8').encode('bz2').encode('base64')
            return pickle.dumps(value, protocol=1).encode('utf-8').encode('bz2').encode('base64')
        elif isinstance(value, np.ndarray):
            #string=pickle.dumps(value.tolist(), protocol=1)
            #return string.encode('utf-8').encode('bz2').encode('base64')
            return pickle.dumps(value.tolist(), protocol=1).encode('utf-8').encode('bz2').encode('base64')
        else:
            raise TypeError('%s is not a list or numpy array' %value)
 
    def value_to_string(self, obj):
        value=self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)

class CompressedNumpyArrayField(models.TextField):
    description='Stores a numpy ndarray in a compressed format (bz2 by default)'

    def __init__(self, *args, **kwargs):
        super(CompressedNumpyArrayField, self).__init__(*args, **kwargs)
    
    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return np.array([])   
        return np.array(pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8'))))
    
    def to_python(self, value):
        if value is None:
            return np.array([])
            
        if isinstance(value, list):
            value=np.array(value)
        if isinstance(value, np.ndarray):
            return value
        
        return np.array(pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8'))))
    
    def get_prep_value(self, value):
        if value is None:
            return value
        elif isinstance(value, list):
            return pickle.dumps(value, protocol=1).encode('utf-8').encode('bz2').encode('base64')
        elif isinstance(value, np.ndarray):
            return pickle.dumps(value.tolist(), protocol=1).encode('utf-8').encode('bz2').encode('base64')
        else:
            raise TypeError('%s is not a list or numpy array' %value)
    
    def value_to_string(self, obj):
        value=self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)

class CompressedDictField(models.TextField):
    description='Stores a python dictionary in a compressed format (bz2 by default)'
    
    def __init__(self, *args, **kwargs):
        super(CompressedDictField, self).__init__(*args, **kwargs)
    
    def from_db_value(self, value, expression, connection, context=None):
        if value is None:
            return {}
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def to_python(self, value):
        if value is None:
            return {}
            
        if isinstance(value, dict):
            return value
        
        return pickle.loads(str(value.decode('base64').decode('bz2').decode('utf-8')))
    
    def get_prep_value(self, value):
        if value is None:
            return value
        elif isinstance(value, dict):
            return pickle.dumps(value, protocol=1).encode('utf-8').encode('bz2').encode('base64')
        else:
            raise TypeError('%s is not a python dictionary' %value)
    
    def value_to_string(self, obj):
        value=self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)
