import os
import numpy as np
import pathlib
import shutil

class Bader:

    def __init__(self, builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):
        from jamip.analysis.vasp.chgcar import Chgcar

        task_id = 'bader'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path
        stdin = pathlib.Path(stdin) 
        stdout = self.rootdir / 'electric' / 'bader'

        # add parameters %
        incar = self.tasks[task_id]
        incar.structure = self.load_structure(stdin)
        self.clear_status(task_id)

        # scf with all charge %
        if (stdin/'AECCAR2').exists():
            chgfile1 = stdin/'AECCAR0'
            chgfile2 = stdin/'AECCAR2'
            if stdout.exists():
                shutil.rmtree(stdout)
            stdout.mkdir(exist_ok=True, parents=True)
            # link CHGCAR
            relpath = os.path.relpath(stdin/"CHGCAR", start=stdout)
            (stdout/"CHGCAR").symlink_to(relpath)
            # link POTCAR
            relpath = os.path.relpath(stdin/"POTCAR", start=stdout)
            (stdout/"POTCAR").symlink_to(relpath)

        else:
            self.calculator(task_id,stdout,stdin)
            chgfile1 = stdout/'AECCAR0'
            chgfile2 = stdout/'AECCAR2'

        # sum charge %
        chgfile3 = stdout/'CHGCAR_sum'
        if chgfile1.exists() and chgfile2.exists():
            chg1 = Chgcar.from_file(chgfile1)
            chg2 = Chgcar.from_file(chgfile2)
            chgsum = chg1.chgcar+chg2.chgcar
            Chgcar.write(chgsum,chg1.poscar,chgfile3)
        
        else:
            raise OSError('AECCAR not exists! bader calculation stop.')

        # run bader %
        os.chdir(stdout)
        os.popen("bader CHGCAR -ref CHGCAR_sum > bader.log").readline()
        os.chdir(self.rootdir)

        # check %
        if (stdout/'ACF.dat').exists():
            status = {'task':'bader','finish':True,'success':True}
            self.write_status(status, stdout)
            incar.state = 'C'
        else:
            incar.state = 'E'

    @classmethod
    def check(self,path):
        path = pathlib.Path(path)
        if path.name == 'bader':
            file = path/'ACF.dat'
        else:
            file = path/'electric'/'bader'/'ACF.dat'
        if file.exists() and file.st_size > 0:
            print('Bader calculation finish.')
            return True
        else:
            return False

    @classmethod
    def load_data(self,path):
        from jamip.analysis.vasp.chgcar import Chgcar
        import pandas as pd
        import re

        path = pathlib.Path(path)
        if path.name == 'bader':
            pass
        else:
            path = path/'electric'/'bader'

        # load POTCAR
        titel = []
        zval = []
        with open(path/"POTCAR") as f:
            for line in f:
                if 'ZVAL' in line:
                    zval.extend(re.findall(r'ZVAL\s*=\s*(\d*\.\d+)',line))
                if 'TITEL' in line:
                    titel.append(line.split('=')[1].split()[1].split('_')[0])

        assert len(titel) == len(zval), f"{titel} not match {zval}"

        # load CHGCAR
        chg = Chgcar.from_file(path/"CHGCAR")
        elements = chg.structure.get_elements(type='symbol')

        baders = []
        with open(path/'ACF.dat', 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                row = line.split()
                if len(row) == 7:
                    baders.append(float(row[4]))
                else:
                    break

        assert len(elements) == len(baders), f"Data not match: Nelements={len(elements)}, Nbaders={len(baders)}"

        zmap = dict(zip(titel,zval))
        zlist = [zmap[i] for i in elements]

        df = pd.DataFrame({'specie':elements, 'zval':zlist, 'bader':baders})
        return df
