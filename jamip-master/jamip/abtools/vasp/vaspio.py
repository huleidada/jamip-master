
__contributor__ = 'Yawen Li, Dianlong Zhao, Xingang Zhao'
#================================================================
# baisc method for control the input and output 
#================================================================

import os
import numpy as np 
import pathlib

class VaspIO(object):

    @classmethod
    def write_poscar(cls, structure, stdout=None, name='POSCAR',\
                          direct=True, vasp5=True, **kwargs):
        """
        'comment','constraints','lattice','elements','numbers','type','positions'
        """
        if stdout is None:  
            stdout = pathlib.Path.cwd()
            path = stdout / name
        else:
            stdout = pathlib.Path(stdout)
            path = stdout if stdout.is_file() else stdout / name
        
        with open(path, 'w') as f:
            # comment line % 
            f.write(str(structure.comment_line)+'\n')

            # scale line %
            f.write('{0:<16.8f}'.format(structure.scale_factor))
            f.write('\n')
            # lattice lines % 
            for l in structure.lattice:
                f.write(' '.join('{0:>16.8f}'.format(c) for c in l))
                f.write('\n')
            # species line % 
            if vasp5 is True:
                f.write(' '.join('{:>6s}'.format(e) for e in structure.species_of_elements))
                f.write('\n')
                
            # number of elements line %
            f.write(' '.join('{0:>6d}'.format(n) for n in structure.number_of_atoms))
            f.write('\n')
    
            # selective dynamics line %
            if structure.select_dynamic is True:
                f.write('Selective Dynamics\n')

            # direct or casterain line % 
            if direct is True:
                f.write('Direct\n')
                # positiion lines %
                for p in structure.atomic_positions:
                    f.write(' '.join('{0:>16.8f}'.format(j) for j in p.scale_coord))
                    if structure.select_dynamic: 
                        f.write('  {0[0]}  {0[1]}  {0[2]}'.format(p.freeze.xyz))
                    f.write('\n')

            elif direct is False:
                f.write('Cartesian\n')
                # positions lines % 
                for p in structure.atomic_positions:
                    f.write(' '.join('{0:>16.8f}'.format(j) for j in p.coord))
                    if structure.select_dynamic: 
                        f.write('  {0[0]}  {0[1]}  {0[2]}'.format(p.freeze.xyz))
                    f.write('\n')

            if structure.initial_velocity is True:
                f.write('\n')
                for p in structure.atomic_positions:
                    f.write(' '.join('{0:>16.8f}'.format(j) for j in p.velocity))
                    f.write('\n')

    @classmethod
    def write_symmetry(cls, structure, stdout=None, name="SYMMETRY",\
                       symprec=1e-5):
        """
        Output the SYMMETY files for calculate the carriers masses 
            by using Boltzmann method.

        args:
            structure:: Structure object contained the methods get_cell()

            stdout:: the output direction of SYMMETY
        """
        import spglib

        if stdout is None:  
            stdout = pathlib.Path.cwd()
            path = stdout / name
        else:
            stdout = pathlib.Path(stdout)
            path = stdout if stdout.is_file() else stdout / name

        dataset = spglib.get_symmetry(structure,symprec=symprec)['rotations']
        sym = []
        for m in dataset:
            m = m.tolist()
            if m not in sym:
                sym.append(m)

        sym.sort()
        sym.reverse()
        with open(path, 'w') as f:

            f.write("%12d" % (len(sym)))

            for rot in sym:
               for k in rot:
                   f.write('\n')
                   f.write(''.join("%10.5f" % v for v in k))
               f.write('\n')

    @classmethod
    def write_potcar(cls, files, stdout, name='POTCAR'):
        """
        Output the POTCAR in the given direction;

        args:
             potcar:: a list include the element potentials; 
             stdout:: the output direction of POTCAR;
        """    
        if stdout is None:  
            stdout = pathlib.Path.cwd()
            path = stdout / name
        else:
            stdout = pathlib.Path(stdout)
            path = stdout if stdout.is_file() else stdout / name

        with open(path, 'w') as f:
            for potcar in files:
                with open(potcar, 'r') as p:
                    f.write(p.read())

    @classmethod
    def write_kpoints(self, kpoints, stdout=None, name='KPOINTS'):
        """
        write out the KPOINTS files if kspacing parameter is None.

        :param kpoints: the object of the kmesh;
        :param stdout: the object path;

        :return: kspacing values if exists.
        """
    
        if stdout is None:  
            stdout = pathlib.Path.cwd()
            path = stdout / name
        else:
            stdout = pathlib.Path(stdout)
            if stdout.is_file():
                path = stdout
                name = stdout.name
            else:
                path = stdout / name
                stdout.mkdir(parents=True, exist_ok=True)

        with open(path,'w') as f:
            f.write('JAMIP KPOINTS\n')

            if kpoints.model == 'Line Model':
                kpoints = kpoints.value

                # get insert
                insert = kpoints.get_insert()
                if name == 'KPOINTS' or name == 'KPOINTS_OPT':

                    if not isinstance(insert, int):
                        raise ValueError('Invalid Kpoints for vasp')
                    
                    f.write(f'{insert}\n')
                    f.write(kpoints.model+'\n')
                    f.write('Direct\n')
                    f.write(repr(kpoints))

                else:
                    f.write(f'{len(kpoints)}\n')
                    f.write('Direct\n')
                    f.write(kpoints.qeformat) 

            elif kpoints.model in ['Gamma','Monkhorst-pack']:
                f.write('0\n')
                f.write(kpoints.model+'\n')
                f.write(' {0[0]} {0[1]} {0[2]}\n'.format(np.array(kpoints.value[0], dtype=int)))
                f.write(' {0[0]} {0[1]} {0[2]}\n'.format(kpoints.value[1]))

            elif kpoints.model == 'Reciprocal':
                f.write(f'{len(kpoints.value)}\n')
                f.write(kpoints.model+'\n')
                for i in kpoints.value:
                    f.write('{0[0]:14.8f}{0[1]:14.8f}{0[2]:14.8f}    {0[3]}\n'.format(i))

        return {}
 
        # automatically produce the kmesh density %
#        if self.auto_density is True:
#
#            density=self.density/sum(structure.num_atoms)
#            
#            direct_cell=np.transpose(structure.cell)
#            rec_cell=2*np.pi*np.transpose(lalg.inv(direct_cell))
#            
#            b1=np.sqrt(np.dot(rec_cell[0],rec_cell[0]))
#            b2=np.sqrt(np.dot(rec_cell[1],rec_cell[1]))
#            b3=np.sqrt(np.dot(rec_cell[2],rec_cell[2]))
 #           
 #           step=(b1*b2*b3/nkpts)**(1./3)
 #           
 #           n1=int(round(b1/step))
 #           if np.mod(n1,2)!=0: n1=n1-1
 #           n2=int(round(b2/step))
 #           if np.mod(n2,2)!=0: n2=n2-1
 #           n3=int(round(b3/step))
 #           if np.mod(n3,2)!=0: n3=n3-1
 #           
 #           if n1==0:n1=1
 #           if n2==0:n2=1
 #           if n3==0:n3=1
 #         
 #       self.comment = 'Auto density {0}/atom'.format(density) 
 #       self.num = 0  
 #       self.__model = 'Gamma'
 #           self.__kpoints = [[n1, n2, n3],[0., 0.,0.]]
 #
 
    @classmethod
    def write_optcell(self, line:str, stdout:str, name='OPTCELL'):

        stdout = pathlib.Path(stdout)
        with open(stdout / name, 'w') as f:
            f.write(line)
 
    @classmethod
    def write_files(self, files:list, stdout:str):
        """Prepare the vdw_kernel""" 
        import shutil

        stdout = pathlib.Path(stdout)
        stdout.mkdir(parents=True, exist_ok=True)

        for file in files:
            if pathlib.Path(file).exists():
                shutil.copy(file, stdout)

    @classmethod
    def write_incar(self, incar, stdout=None, **kwargs):
        """
        function: output the INCAR file with format: key = value.
        :param name: output file name, type: string
        :param parses: INCAR input parameters. type: dict
        :param args: exclude parameter.
        :param kwargs: external parameter.
        """
        from .vaspflow import Task
        from copy import deepcopy

        if isinstance(incar, dict):
            incar = Task(incar)
        elif isinstance(incar, Task):
            incar = deepcopy(incar)
        else:
            raise ValueError('Type of incar params should be dict or Incar!')

        # update incar %
        incar.update(kwargs)
        if hasattr(incar,'exclude'):
            for key in incar.exclude:
                incar.pop(key)

        # format information %
        lines = ''
        #for key in reversed(incar.group):
        for key in incar.group:
            if not isinstance(incar.group[key], list): continue
            # exclude params
            if key == 'exclude':
                for i in incar.group[key]:
                    if i in incar: 
                        incar.pop(i)
                continue

            lines += '\n# %s Parameters\n' %key.upper()
            for i in incar.group[key]:
                if i in incar: 
                    lines += "%-15s = %-s\n" %(i.upper(),incar.pop(i))


        with open(os.path.join(stdout,'INCAR'), 'w') as f:

            for key,value in incar.items():
                if value != None and value != '':
                    f.write("%-15s = %-s\n" %(key.upper(),value))
            f.write(lines)
