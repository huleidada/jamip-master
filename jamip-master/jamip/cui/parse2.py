import argcomplete, argparse
import pathlib


class __ParseProcess(object):

    def __collect_parses__(self):
    
        parser = self.__get_parse__()
        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        want_json = getattr(args, 'emit_json', False)
        # (options, args) =self.__get_parse__().parse_args()
        #print(args)
 
        all_option = {}       # parameters %

        if args.pool:
            all_option['pool'] = args.pool
        if args.abtools:
            all_option['soft'] = args.abtools

        # # fast data processing %
        if args.output:
            all_option['output'] = args.output
            if want_json:
                all_option['json'] = True
            return all_option

        # # vasp_tools %
        if args.vasp_tools:
            all_option['vasp_tools'] = args.vasp_tools
            if want_json:
                all_option['json'] = True
            return all_option

        # # cp2k_tools %
        if args.cp2k_tools:
            all_option['cp2k_tools'] = args.cp2k_tools
            if want_json:
                all_option['json'] = True
            return all_option

        # #input examples % 
        if args.input:
            all_option['input'] = args.input
            if want_json:
                all_option['json'] = True
            return all_option
	
        # cluster environments % 
        cluster = {}
        if args.cores:
            cluster['core'] = args.cores 
        if args.nodes:
            cluster['node'] = args.nodes
        if args.queue:
            cluster['queue'] = args.queue 	
        if args.maxjobs:
            all_option['maximum'] = args.maxjobs
        if len(cluster) >0:
             all_option['cluster'] = cluster
    
        # #run/output the required files for calc % 
        if args.submit:
            all_option['run'] = args.submit
        if want_json:
            all_option['json'] = True

        if args.phonon:
            all_option['phonon'] = args.phonon
    
        if args.mysql:
            all_option['mysql'] = args.mysql
        if args.django:
            all_option['django'] = args.django
    
        # #check the status %
        if args.check:
            all_option['check'] = args.check

        if hasattr(args,'job'):
            all_option['plot'] = args.job
        if hasattr(args,'soft'):
            all_option['soft'] = args.soft
        if hasattr(args,'database'):
            all_option['db'] = args.database
    
        return all_option


    
    def __get_parse__(self):
        import jamip

        parser = argparse.ArgumentParser(prog="JAMIP", description='jamip [options] args')
        parser.add_argument('--version', action='version', version=f'%(prog)s {jamip.__version__}')

        # parser for sub commands 
        subparsers = parser.add_subparsers(help='sub-command help')

        parser_plot = subparsers.add_parser('plot', help='plot help')

        parser_plot.add_argument(dest='job', action='store', choices=['band','dos','absorb'], 
                                 nargs=1, default=None, help='Present a basic example for calculation.\n')
        parser_plot.add_argument('-s','--soft', dest='soft', action='store', choices=['vasp','qe','phonopy'],
                                 nargs=1, default='vasp', help='Sets the software to use for plotting\n')
        
        parser_db = subparsers.add_parser('db', help='plot help')
        parser_db.add_argument(dest='database', action='store', type=str,
                         choices=['entry','structure','history'], 
                         default=None, help='Interacting with Databases\n')
    
        # parser for default commands
        group = parser.add_mutually_exclusive_group()
    
        group.add_argument('-i', '--input', dest='input', action='store', 
                         metavar='FILE', default=None, choices=['vasp', 'win2k', 'abinit', 'qe', 'gaussian','cp2k','plot'], 
                         help='Present a basic example for calculation.\n')

        group.add_argument('-r', '--run', dest='submit', choices=['input', 'qsub', 'prepare','single','skip'],
                         default=None, help='Submit the tasks according to the input data.\n')    

        group.add_argument('-o', '--output', dest='output', action='store', nargs='+',
                         default=None, help='Output data from calculated result.\n')

        group.add_argument('-v', '--vasp', dest='vasp_tools', action='store', 
                            choices=['potcar', 'bond', 'standard', 'kpath', 'kpath2d', 'clean', 'dim'],
                         default=None, help='simple vasp tools for baise calculationst.\n')

        group.add_argument('--cp2k', dest='cp2k_tools', action='store', 
                            choices=['poscar', 'bond', 'kpath', 'kpath2d', 'clean', 'dim'],
                         default=None, help='simple vasp tools for baise calculationst.\n')
        
        group.add_argument('-c', '--check', dest='check', action='store', 
                         default=False, choices=['show','prepare','status','qstat','bjobs','squeue','converge','reduce'],
                         help='Check status of tasks.\n')


        #group_extract.add_option('-t', '--tar', dest='compress', action='store_true', 
        #                 default=False, help='backup the data to a tarfile named xx.tar.bz2\n')

        group.add_argument('--phonon', dest='phonon',action='store',type=str,
                         choices=['fc2','fc3','raman','help','gruneisen','band'],default=None,
                         help='Phonopy module.\n')

        group.add_argument('--mysql', dest='mysql',action='store',type=str,
                         choices=['initialize','start','shutdown'],default=None,
                         help='MySQL management module.\n')

        group.add_argument('--django',dest='django',action='store',type=str,
                         choices=['mysql','sqlite','makemigrations','migrate','dumpdata','loaddata','flush'],default=None,
                         help='Django management module.\n')

        parser.add_argument('-f', '--file', dest='pool', action='store', nargs='+',
                         type=pathlib.Path, default=None, help='Pool name for storing the calculating data.\n')

        parser.add_argument('--json', dest='emit_json', action='store_true', default=False,
                         help='Machine-readable JSON on stdout: prepare（任务池）; check qstat/squeue/bjobs（调度器作业）; check show（本地池步骤状态）。prepare 时抑制相关 INFO。\n')

        parser.add_argument('--soft', dest='abtools', action='store',
                         type=str, default=None, help='Abtools for calculating data.\n')

        cluster = parser.add_argument_group('cluster', 'cluster settings')

        cluster.add_argument('--queue', dest='queue', action='store', 
                         type=str, default=None, help='Which queue for projecting tasks.\n')
        cluster.add_argument('--cores', dest='cores', action='store', 
                         type=int, default=None, help='How many codes to be used for one task.\n')    
        cluster.add_argument('--num', dest='maxjobs', action='store', 
                         type=int, default=None, help='The maxmium number of jobs to be submit at once.\n')   
        cluster.add_argument('--nodes', dest='nodes', action='store', 
                         type=int, default=None, help='How many cluster to be used.\n')

        #group_cluster.add_option('--ssh', dest='sshlog', action='store', type='string', 
        #                 default=None, help='remote manager the tasks, using --ssh=username@192.168.1.1:port\n')


        return parser

