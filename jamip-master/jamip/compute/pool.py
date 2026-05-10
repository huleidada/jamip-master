# coding: utf-8
# Copyright (c) JAMIP Development Team.
# Distributed under the terms of the JLU License.

#=================================================================
# This file is part of JAMIP
#
# Copyright (C) 2021 Jilin University
#
#  JAMIP is a platform for high throughput calculation. It aims to 
#  make simple to organize and run large numbers of tasks on the 
#  superclusters and post-process the calculated results.
#  
#  JAMIP is a useful packages integrated the interfaces for ab initio 
#  programs, such as, VASP, Guassian, QE, Abinit and 
#  comprehensive workflows for automatically calculating by using 
#  simple parameters. Lots of methods to organize the structures 
#  for high throughput calculation are provided, such as alloy,
#  heterostructures, etc.The large number of data are appended in
#  the MySQL databases for further analysis by using machine 
#  learning.
#
#  JAMIP is free software. You can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published 
#  by the Free sofware Foundation, either version 3 of the License,
#  or (at your option) and later version.
# 
#  You should have recieved a copy of the GNU General Pulbic Lincense
#  along with JAMIP. If not, see <https://www.gnu.org/licenses/>.
#=================================================================

"""
This module defines the classes relating to tasks pool.
"""

__author__ = "Kun Zhou"
__copyright__ = "Copyright 2021, The JAMIP"
__version__ = "1.3"
__maintainer__ = "JAMIP team"
__email__ = "zhoukun21@mails.jlu.edu.cn"
__status__ = "underdeveloped"

from collections import UserDict
import logging
import os
import pathlib
import json
import lmdb
import time
import threading
from contextlib import contextmanager

class PoolDict(UserDict):
    """并发安全的键值存储"""

    def __init__(self, db_file='safe_kv.db', mode='r', map_size=10*1024*1024*1024, max_readers=126):
        self.db_file = db_file
        self.map_size = map_size
        self.max_readers = max_readers
        self._local = threading.local()
        self._init_db(mode)

    def _init_db(self, mode):
        """Initialize database"""
        db_path = pathlib.Path(self.db_file)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.env = lmdb.open(
            str(db_path),
            map_size=self.map_size,
            max_readers=self.max_readers,
            max_dbs=1,
            lock=True,  # 需要锁以支持并发写
            sync=True,  # 同步写入，确保数据安全
            metasync=True,  # 同步元数据
            readahead=False,  # NFS上禁用预读
            writemap=False,  # NFS上禁用写映射
            map_async=False  # 异步映射
        )

        if mode == 'n':
            self.clear()

    @contextmanager
    def _get_txn(self, write=False):
        """获取带重试的数据库连接"""
        if not write and hasattr(self._local, 'read_txn'):
            yield self._local.read_txn
        else:
            txn = self.env.begin(write=write)
            try:
                yield txn
                if write:
                    txn.commit()
            except Exception:
                if write:
                    txn.abort()
                raise
            finally:
                if not write:
                    txn.abort()  # 读事务必须abort
                elif not write and hasattr(self._local, 'read_txn'):
                    delattr(self._local, 'read_txn')                

    def _serialize(self, value):
        """序列化值（支持字符串和整数）"""
        return json.dumps(value).encode()
    
    def _deserialize(self, data):
        """反序列化值"""
        if data is None:
            return None
        return json.loads(data.decode())                    

    def __setitem__(self, key, value):
        """Support dict[key] = value """
        with self._get_txn(write=True) as txn:
            txn.put(key.encode(), self._serialize(value), overwrite=True)        

    def __getitem__(self, key):
        """Support value = dict[key] """
        with self._get_txn(write=False) as txn:
            data = txn.get(key.encode())
            if data is None:
                raise KeyError(f"Key not exists: {key}")
            return self._deserialize(data)

    def __delitem__(self, key):
        """Support del dict[key] """
        with self._get_txn(write=True) as txn:
            if not txn.delete(key.encode()):
                raise KeyError(f"Key not exists: {key}")

    def __len__(self):
        """Support len(dict) """
        with self._get_txn(write=False) as txn:
            return txn.stat()['entries']        

    def __contains__(self, key):
        """Support key in dict """
        with self._get_txn(write=False) as txn:
            return txn.get(key.encode()) is not None        

    def keys(self):
        """Support dict.keys() """
        with self._get_txn(write=False) as txn:
            with txn.cursor() as cursor:
                return [key.decode() for key, _ in cursor]

    def values(self):
        """Support dict.values() """
        with self._get_txn(write=False) as txn:
            with txn.cursor() as cursor:
                return [self._deserialize(value) for _, value in cursor]        

    def items(self):
        """Support dict.items() """
        with self._get_txn(write=False) as txn:
            with txn.cursor() as cursor:
                return [(key.decode(), self._deserialize(value)) 
                        for key, value in cursor]

    def get(self, key, default=None):
        """Support dict.get(key, default) """
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, default=None):
        """Support dict.pop(key, default) """
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is not None:
                return default
            raise

    def clear(self):
        """Support dict.clear() """
        with self._get_txn(write=True) as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    cursor.delete()        

    def __iter__(self):
        """Support: for key in dict """
        return iter(self.keys())

    def __str__(self):
        """Support: print(dict) """
        items = []
        for key, value in self.items():
            # 限制显示长度
            if len(str(value)) > 20:
                value_str = str(value)[:17] + '...'
            else:
                value_str = str(value)
            items.append(f"'{key}': '{value_str}'")

        return '{' + ', '.join(items) + '}'

    def close(self):
        """Release LMDB handles so the same path can be reopened in-process."""
        env = getattr(self, 'env', None)
        if env is None:
            return
        try:
            env.sync()
        except Exception:
            pass
        try:
            env.close()
        finally:
            self.env = None

    def update(self, other=None, **kwargs):
        """Support dict.update() """
        with self._get_txn(write=True) as txn:
            if other:
                if hasattr(other, 'items'):
                    items = other.items()
                else:
                    items = other
                
                for key, value in items:
                    txn.put(key.encode(), self._serialize(value), overwrite=True)
            
            for key, value in kwargs.items():
                txn.put(key.encode(), self._serialize(value), overwrite=True)        

    def update_status(self, path, status='C'):

        tmp = self[path]
        _status = tmp['status']
        tmp['status'] = status
        self[path] = tmp
        return _status

    def update_jobid(self, path, value:int):

        tmp = self[path]
        tmp['id'] = value
        tmp['prior'] -= 1
        tmp['status'] = 'R'
        self[path] = tmp

class Pool:
      
    """
    任务池是存储计算输入数据，并将根据任务提交请求提交和记录任务状态。
    任务池分为数据部分(jp.pool)和记录部分(jp.dat/jp.dir/jp.bak)
    其中数据部分基于pickle，记录部分基于dbm
    任务池整体支持轻量级的并行访问，安全性有待测试，可能通过冗余的方式解决

    Atributes:
        _pool: Calculation data saved in *.pool
        __db: Work status of jobs, saved in *.dat

    Functions:
        pass
    """
    class open:

        def __init__(self,filepath,mode='r'):
            filepath = pathlib.Path(filepath).absolute()
            self.filepath = filepath.parent / f'.{filepath.name}'
            self.mode=mode
            # self.encoding=encoding
 
        def __enter__(self):
            self.kv = PoolDict(self.filepath, self.mode)
            return self.kv
 
        def __exit__(self, *args):
            if getattr(self, 'kv', None) is not None:
                self.kv.close()
                self.kv = None

    def __init__(self, pool=None, **kwargs):

        self._pool = {}
        self.__functional = None
        self.__structure = None
        self.__db = None
        self.poolname = pool

    @property
    def joblist(self):
        return self._pool.keys()

    @property
    def pool(self):
        return self.__db

    def add_tasks(self, path, func):
        from os import getcwd
        from os.path import relpath, normpath, abspath
        from copy import deepcopy
	
        rpath = relpath(abspath(normpath(path)), getcwd())
        self._pool[rpath] = deepcopy(func) 

    @property
    def functional(self):
        """
        Object of program, type ModuleFactory.
        """
        return self.__functional

    @functional.setter
    def functional(self, value):
        """
        note: now only VASP object is avaliable
        """
        self.__functional = value

    @property
    def mainkey(self):
        maink = {}
        for key,value in self._pool.items():
            maink[key] = value.structure.get_formula()

        return maink
            
    def save(self, path=None, mode='n', **kwargs):
        """
        Method aims to store input information in a binary file.

        args:
            path
            pool_name:: string, name of output file;
            overwrite:: bool, overwrite the file or not;        
        """

        from filelock import SoftFileLock
        import pickle
        import fcntl

        path = pathlib.Path(path).absolute()
        # directoty check %
        if not path.parent.exists():
            path.parent.mkdir()

        try:
            with open(path, 'wb+') as f:
                # lock file %
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
             
                # update pool %
                pool = pickle.load(f) if path.stat().st_size else {}         
                if mode == 'n':          # overwrite %
                    pool = self._pool    
                else:                       # update %
                    pool.update(self._pool) 
                pool.update(kwargs)
             
                # rewrite file and unlock %
                f.truncate()
                pickle.dump(pool, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except:
            lockfile = path.parent / f'.{path.name}.lock'
            with SoftFileLock(lockfile):
                with open(path, 'wb+') as f:

                    # update pool %
                    pool = pickle.load(f) if path.stat().st_size else {}

                    if mode == 'n':          # overwrite %
                        pool = self._pool
                    else:                       # update %
                        pool.update(self._pool)
                    pool.update(kwargs)

                    # rewrite file and unlock %
                    f.truncate()
                    pickle.dump(pool, f)
        finally:
            if os.environ.get('JAMIP_PREPARE_JSON') != '1':
                logging.info(f'Pool save in {path}')
            del pool

        return self

    def merged(self, path=None, *args, **kwargs):
        """
        merge the files exists pool files
        """
        self.save(path, False, *args)
        return self

    @classmethod
    def loader(cls, path:str, *args, **kwargs):
        """
        load the pool with only one proccess %
        """
        from jamip.utils.logger import full_path
        import pickle
        #import fcntl

        fpool = full_path(path)

        # load pool %
        with open(fpool, 'rb') as f:
            # fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            pool = pickle.load(f)
            if pool == None or len(pool) == 0:
                logging.warning('Empty Pool : %s' %path)

        return pool

    def close(self):
        """
        close pool and pool-db
        """
        import time
        for i in range(6):
            try: 
                self.__db.sync()
                logging.info('Pool Update Finish.')
                self.__db.close()
                return 
            except:
                logging.warning('Pool sync block !')
                time.sleep(10)
  
        logging.error('Pool sync Failed.')
        exit()

def get_pool_list(path):        
    import dbm
    pool_list = []
    root = pathlib.Path(path).parent
    with Pool.open(path) as p:
        pool_list = [root/outdir for outdir in p]
    return pool_list
        

