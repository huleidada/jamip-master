import os
import time
import shutil

# check if MySQL installed %
mysql_search = os.popen('conda list|grep mysql').readlines()
mysql_version = None
for line in mysql_search:
    if line.split()[0] == 'mysql':
        mysql_version = line.split()[1]

if mysql_version is None:
    print('MySQL not installed !')
    os.sys.exit()

# version warning %
print('Your mysql version :',mysql_version)
if not mysql_version.startswith('5.7'):
    print('Warning! Developers using the 5.7.24 version of MySQL, you may have problems in the software installation.')

# set mysql_data directory %
mysql_dir = os.environ['HOME']+'/mysql'
if os.path.exists(mysql_dir):
    print('%s already exists. Do you want to delete and regenerate it? (Y/N)' %mysql_dir)
    d1 = input('Please input: ')
    if d1.lower() == 'y':
        shutil.rmtree(mysql_dir) 
        os.makedirs(mysql_dir)
    else:
        os.sys.exit()
else:
    os.makedirs(mysql_dir)

# set port %
port = None
while port is None:
    port = input('set mysql port: ')
    port_serach = os.popen('netstat -alt|grep %s' %port).readline()
    if port_serach:
        print('This port is occupied. Please try another port')
        port = None

# set lc-messages-dir %
lc_dirs = os.popen('ls ~/anaconda3/pkgs/ |grep mysql').readline().rstrip()
lc_dirs = os.path.join(os.environ['HOME']+'/anaconda3/pkgs/%s/share/' %lc_dirs)
if not os.path.exists(lc_dirs):
    print('Automatic find lc-messages-dir failed. Please try entering the path manually.')
    os.sys.exit()

# write my.cnf %
with open(mysql_dir+'/my.cnf','w') as f:
    f.write('[client]\nport={1}\nsocket={0}\mysql.sock\ncharacter-set-server=utf8\n\n[mysqld]\nport={1}\nbasedir={0}\ndatadir={0}/data\npid-file={0}/mysql.pid\nsocket={0}/mysql.sock\nlog_error={0}/error.log\nlc-messages-dir={2}\nlc-messages=en_US\ncharacter-set-server=utf8\nserver-id=100\n'.format(mysql_dir,port,lc_dirs))

# create other files %
os.makedirs(mysql_dir+'/data')
open(mysql_dir+'/error.log',"w+").close()
open(mysql_dir+'/mysql.pid',"w+").close()
open(mysql_dir+'/mysql.sock',"w+").close()

# create softlink %
try:
    default_sock = '/tmp/mysql.sock' 
    if os.path.lexists(default_sock):
        os.remove(default_sock)
    user_sock = mysql_dir+'/mysql.sock'
    os.symlink(user_sock,default_sock)
except:
    print('Automatic create softlink failed. Please enter "ln -s ~/mysql/mysql.sock /tmp/mysql.sock" manually.')
    os.sys.exit()
    
print("File ready to complete, start initialization.")
# mysqld initialize %
os.popen("mysqld --defaults-file=%s/my.cnf --initialize" %mysql_dir).readline()

# get password %
while True:
    status = os.popen("lsof %s/error.log" %mysql_dir).readline()
    if len(status) == 0: 
        passwd_line = os.popen("tail -1 %s/error.log" %mysql_dir).readline()
        break
    time.sleep(1)
    
if "password" in passwd_line:
    passwd = passwd_line.split()[-1]
    print("Mysql initialization succeeded. Enter 'mysqld_safe --defaults-file=%s/my.cnf &' to start MySQL." %mysql_dir)
    print("Enter 'mysql -uroot -p' to start your first login ^-^")
    print("Your password is %s" %passwd)
else:
    print('Failed initialization. Sorry.')



