import os
import numpy as np
from pathlib import Path
import shutil
from jamip.structure import read,write,Structure

class Shengbte:

    def __init__(self, builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):
        from jamip.analysis.vasp.md import TDEP

        task_id = 'shengbte'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = Path(self.tasks[self.links[task_id][0]].path)
        stdout = self.rootdir / 'md' / 'shengbte'
        incar = self.tasks[task_id]
        self.clear_status(task_id)

        stdout.mkdir(exist_ok=True, parents=True)
        # calculate base TDEP structures %
        if (stdin/"infile.ssposcar").exists():
            shutil.copy(stdin/"infile.ssposcar", stdout/"infile.ssposcar")
            shutil.copy(stdin/"infile.ucposcar", stdout/"infile.ucposcar")
            self.write_control_with_tdep(incar, stdout)

        # convert tdep_fc to shengbte
        tdep = TDEP(workdir=stdin)
        tdep.toPhonopy(stdout,filename="FORCE_CONSTANTS_2ND")
        tdep.toShengBTE(stdout)

        os.chdir(stdout)
        os.environ['NUMEXPR_MAX_THREADS'] = '16'
        incar.program = "ShengBTE"
        self.cluster.run(incar, "shengbte.log")
        os.chdir(self.rootdir)

        if (stdout/"shengbte.log").exists():
            with open(stdout/"shengbte.log", 'r') as f:
                lines = f.readlines()
            if len(lines) and "normal exit" in lines[-1]:
                status = {'task':'shengbte','finish':True,'success':True}
                self.write_status(status, stdout)
                incar.state = 'C'

    @classmethod
    def check(self,path):
        path = Path(path)
        if path.name == 'shengbte':
            file = path/'shengbte.log'
        else:
            file = path/'md'/'shengbte'/'shengbte.log'
        if file.exists() and file.st_size > 0:
            with open(file, 'r') as f:
                lines = f.readlines()
            if len(lines) and "normal exit" in lines[-1]:
                print('ShengBTE calculation finish.')
                return True

        return False

    def write_control_with_tdep(self, incar, stdin):

        #read structral information from infile.ucposcar
        uc = read(stdin/"infile.ucposcar", ftype='vasp')
        ss = read(stdin/"infile.ssposcar", ftype='vasp')

        ngrid = incar.get("ngrid", "10 10 10")
        dim = incar.get("dim", "2 2 2")
        T = incar.get("T", 300)
        T_min = incar.get("T_min", 300)
        T_max = incar.get("T_max", 300)
        T_step = incar.get("T_step", 300)
        scalebroad = incar.get("scalebroad", 0.1)
        
        with open(stdin/"CONTROL",mode='w') as f:
            f.write("&allocations\n")
            f.write(f"  nelements={len(uc.species_of_elements)}\n")
            f.write(f"  natoms={len(uc)}\n")
            f.write(f"  ngrid(:)={ngrid}\n")
            f.write("&end\n")

            f.write("&crystal\n")
            f.write(f"  lfactor={uc.scale_factor/10}\n")
            lat = uc.lattice
            for i in range(0,3):
                f.write(f"  lattvec(:,{i+1})= {lat[i,0]:10.6} {lat[i,1]:10.6} {lat[i,2]:10.6}\n")
        
            speciesline = ' '.join('{:>6s}'.format(e) for e in uc.species_of_elements)
            f.write(f"  elements={speciesline}\n")

            numbersline = ' '.join(np.repeat(np.arange(1,len(uc.number_of_atoms)+1), uc.number_of_atoms).astype(str))
            f.write(f"  types={numbersline}\n")

            for i,row in enumerate(uc.atomic_positions):
                coordsline = ' '.join('{0:>16.8f}'.format(j) for j in row.scale_coord)
                f.write(f"  positions(:,{i+1})={coordsline}\n")
        
            f.write(f"  scell(:)= {dim}\n")
            f.write("&end\n")
            f.write("&parameters\n")
            f.write(f"  T={T}\n")
            f.write(f"  scalebroad={scalebroad}\n")
            f.write("&end\n")
            f.write("&flags\n")
            f.write("  nonanalytic=.TRUE.\n")
            f.write("  nanowires=.FALSE.\n")
            f.write("&end")
        
