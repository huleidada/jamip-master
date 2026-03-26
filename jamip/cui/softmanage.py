import os

class __Mysql(object):

    def __init__(self, params, *args, **kwargs):
        
        if params['mysql']  == 'initialize':
            os.system('python ~/.jamip/bin/mysql_init.py')
        elif params['mysql'] == 'start':
            os.system('mysqld_safe --defaults-file=~/mysql/my.cnf &')
        elif params['mysql'] == 'shutdown':
            os.system('mysqladmin -uroot -p shutdown')

class __Django(object):

    def __init__(self, params, *args, **kwargs):
        

        if params['django']  == 'mysql':
            self.set_mysql_default()
        elif params['django']  == 'sqlite':
            self.set_sqlite_default()
        elif params['django']  == 'postgresql':
            self.set_postgresql_default()
        elif 'pool' in params:
            self.django_manage(params['django'], params['pool'])
        else:
            self.django_manage(params['django'])

    def set_sqlite_default(self):
        import json
        basedir = os.path.join(os.environ['HOME'],'.jamip','bin','jamipdb')
        default = {'ENGINE': 'django.db.backends.sqlite3',
                   'NAME': basedir}
        with open(os.environ['HOME']+'/.jamip/env/django.json','w') as f:
            f.write(json.dumps(default,indent=3))
        
    def set_mysql_default(self):
        import json
        import socket
        print('SET DJANGO DATABASE DEFAULT')
        default = {'ENGINE': 'django.db.backends.postgresql',
                   'HOSTNAME': socket.gethostname(),
                   } 
        default['NAME'] = input('Database name: ')
        default['USER'] = input('PostgreSQL username: ')
        default['PASSWORD'] = input('PostgreSQL password: ')
        default['HOST'] = input('Cluster IP: ')
        default['PORT'] = input('PostgreSQL port: ')
        with open(os.environ['HOME']+'/.jamip/env/django.json','w') as f:
            f.write(json.dumps(default,indent=3))

    def set_mysql_default(self):
        import json
        import socket
        print('SET DJANGO DATABASE DEFAULT')
        print("Do you use a Mysql database and complete the installation based on JAMIP?")
        if input('Enter (Y/N): ').lower() == 'y':
            sock = os.environ['HOME']+'/mysql/mysql.sock'
        else:
            print("Please manually modify the configuration file in '~/.jamip/env/django.json'")
            os.sys.exit()
        default = {'ENGINE': 'django.db.backends.mysql',
                   'HOSTNAME': socket.gethostname(),
                   'OPTIONS': {'unix_socket':sock}
                   } 
        default['NAME'] = input('Database name: ')
        default['USER'] = input('Mysql username: ')
        default['PASSWORD'] = input('Mysql password: ')
        default['HOST'] = input('Cluster IP: ')
        default['PORT'] = input('Mysql port: ')
        with open(os.environ['HOME']+'/.jamip/env/django.json','w') as f:
            f.write(json.dumps(default,indent=3))

    def django_manage(self,argv,args=None):
        from .script import get_package_path
        argvs = ['manage.py',argv]
        if argv == "makemigrations":
            root = get_package_path()
            dir = os.path.join(root,'db','materials','migrations')
            for file in os.listdir(dir):
                if file.startswith('00'):
                    os.remove(os.path.join(dir,file))
            db = os.path.join(os.environ['HOME'],'.jamip','bin','jamipdb')
            if os.path.exists(db):
                os.rename(db,db+'.bk')

        elif argv == 'dumpdata':
            if args != None:
                argvs.append(args)

        elif argv == 'loaddata':
            if args == None:
                raise Exception("Please add json filename.")
            else:
                argvs.append(args)

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jamip.db.db.settings")
        try:
            from django.core.management import execute_from_command_line
        except ImportError:
            # The above import may fail for some other reason. Ensure that the
            # issue is really that Django is missing to avoid masking other
            # exceptions on Python 2.
            try:
                import django
            except ImportError:
                raise ImportError(
                    "Couldn't import Django. Are you sure it's installed and "
                    "available on your PYTHONPATH environment variable? Did you "
                    "forget to activate a virtual environment?"
                )
            raise

        if argv == 'dumpdata':        
            import sys
            savedStdout = sys.stdout 
            with open('jamip.json', 'w') as file:
                sys.stdout = file 
                execute_from_command_line(argvs)
            sys.stdout = savedStdout
        else:
            execute_from_command_line(argvs)
