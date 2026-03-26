# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os


class Write(object):
    """
    write a structure file. At present, the code support the following file types: cif, poscar, xyz, mol.

    """
    def __init__(self, structure, path, dtype=None, **kwargs):
        """
        Arguments:
            structure: object of structure.
            python: path of file. i.e. /home/xx/xx/POSCAR
            dtype: data type. 
                crystal: cif, poscar
                molecule: xyz, mol
                
            kwargs:
                for 'poscar' type:
                    coordinate_type (default='Direct'): type of atomic coordinate ('Direct' or 'Cartesian').
                    isContainedConstraints (default=False): whether to contain the atomic constraint information.
                    isContainedVelocities (default=False): whether to contain the atomic velocity information.
        """
        from ..materials.structure import Structure
        from ..materials.molStructure import MolStructure
        
        self.structure=structure
        self.path=path
        
        if dtype == None:
            if os.path.basename(path).lower().endswith('.cif'):
                self.dtype='cif'
                if not isinstance(structure, Structure):
                    raise ValueError('structure is not belong to the Structure')
            elif os.path.basename(path).lower().endswith('.vasp') or \
                os.path.basename(path).lower().endswith('poscar') or \
                os.path.basename(path).lower().endswith('contcar'):
                self.dtype='poscar'
                if not isinstance(structure, Structure):
                    raise ValueError('structure is not belong to the Structure')
            elif  os.path.basename(path).lower().endswith('.xyz'):
                self.dtype='xyz'
                if not isinstance(structure, MolStructure):
                    raise ValueError('structure is not belong to the MolStructure')
            elif os.path.basename(path).lower().endswith('.mol'):
                self.dtype='mol'
                if not isinstance(structure, MolStructure):
                    raise ValueError('structure is not belong to the MolStructure')
            else:
                raise TypeError('unrecognized type')
        elif dtype.lower() == 'cif':
            self.dtype='cif'
        elif dtype.lower() == 'vasp' or \
            dtype.lower() == 'poscar' or \
            dtype.lower() == 'contcar':
            self.dtype='poscar'
        elif dtype == 'xyz':
            self.dtype='xyz'
        elif dtype == 'mol':
            self.dtype='mol'
        else:
            raise TypeError('unrecognized type')
        
        # for poscar
        if self.dtype == 'poscar':
            self.coordinate_type='Direct'
            if 'coordinate_type' in kwargs:
                self.coordinate_type=kwargs['coordinate_type']
                if not(self.coordinate_type.strip().lower().startswith('d')) and not(self.coordinate_type.strip().lower().startswith('c')):
                    raise ValueError('unrecognized type of atomic coordinate in coordinate_type')
                
            self.isContainedConstraints=False
            if 'isContainedConstraints' in kwargs:
                self.isContainedConstraints=kwargs['isContainedConstraints']
                if not isinstance(self.isContainedConstraints, bool):
                    raise ValueError('isContainedConstraints is not a boolean value')
            
            self.isContainedVelocities=False
            if 'isContainedVelocities' in kwargs:
                self.isContainedVelocities=kwargs['isContainedVelocities']
                if not isinstance(self.isContainedVelocities, bool):
                    raise ValueError('isContainedVelocities is not a boolean value')
                
    def run(self):
        """
        write file, output structure.
        
        """
        if self.dtype == 'cif':
            self.__writeCIF()    
        elif self.dtype == 'poscar':
            self.__writePOSCAR(self.coordinate_type, self.isContainedConstraints, self.isContainedVelocities)
        elif self.dtype == 'xyz':
            self.__writeXYZ()
        elif self.dtype == 'mol':
            self.__writeMOL()
        
    def __writeCIF(self):
        """
        write CIF.
        
        """
        from collections import defaultdict
        cif=self.structure.formatting(dtype='cif')
        
        output=open(self.path, 'w')
        
        #data_
        output.write('data_'+cif['data_']+'\n')
        output.write('#By JAMIP\n')

        #lattice
        output.write('_cell_length_a %32.4f\n' %(cif['_cell_length_a']))
        output.write('_cell_length_b %32.4f\n' %(cif['_cell_length_b']))
        output.write('_cell_length_c %32.4f\n' %(cif['_cell_length_c']))
        output.write('_cell_angle_alpha %29.4f\n' %(cif['_cell_angle_alpha']))
        output.write('_cell_angle_beta %30.4f\n' %(cif['_cell_angle_beta']))
        output.write('_cell_angle_gamma %29.4f\n' %(cif['_cell_angle_gamma']))

        #volume
        output.write('_cell_volume %34.4f\n' %(cif['_cell_volume']))

        #spacegroup
        output.write('_symmetry_space_group_name_H-M %15s\n' %(cif['_symmetry_space_group_name_H-M']))
        output.write('_symmetry_Int_Tables_number %17s\n' %(cif['_symmetry_Int_Tables_number']))
            
        #symmetry
        equnum=0
        output.write('loop_\n')
        output.write('_symmetry_equiv_pos_as_xyz\n')
        for equxyz in cif['_symmetry_equiv_pos_as_xyz']:
            output.write('    '+equxyz+'\n')

        #atom site
        output.write('loop_\n')
        output.write('_atom_site_label\n')
        output.write('_atom_site_type_symbol\n')
        output.write('_atom_site_symmetry_multiplicity\n')
        output.write('_atom_site_wyckoff_symbol\n')
        output.write('_atom_site_fract_x\n')
        output.write('_atom_site_fract_y\n')
        output.write('_atom_site_fract_z\n')
        output.write('_atom_site_occupancy\n')
        
        atom_label=[]
        atom_symbol=cif['_atom_site_label']
        counter=defaultdict(int)
        for symbol in atom_symbol:
            counter[symbol] += 1
            atom_label.append('%s%d' %(symbol,counter[symbol]))

        atom_multi=cif['_atom_site_symmetry_multiplicity']
        wyckoffs=cif['_atom_site_wyckoff_symbol']
        coords=cif['_atom_site_fract']
        occs=cif['_atom_site_occupancy']
        assert len(atom_symbol) == len(atom_multi) == len(wyckoffs) == len(coords) == len(occs)
            

        for i,coord in enumerate(coords):
            output.write('{}    {}   {}   {}   '.format(atom_label[i], atom_symbol[i], atom_multi[i], wyckoffs[i]))
            output.write('%16.8f %16.8f %16.8f %8.2f\n' %(coord[0],coord[1],coord[2],occs[i]))
            
        output.close()
    
    def __writePOSCAR(self, coordinate_type, isContainedConstraints, isContainedVelocities):
        """
        write POSCAR.
        
        Arguments:
            coordinate_type: type of atomic coordinate ('Direct' or 'Cartesian').
            isContainedConstraints: whether to contain the atomic constraint information.
            isContainedVelocities: whether to contain the atomic velocity information.
        """        
        poscar={}
        
        poscar=self.structure.formatting(dtype='poscar', 
                                         coordinate_type=coordinate_type, 
                                         isContainedConstraints=isContainedConstraints, 
                                         isContainedVelocities=isContainedVelocities)
      
        output=open(self.path, 'w')
        
        output.write(poscar['comment']+'\n')
            
        output.write('   1.0'+'\n') # scale value. Note that: don't modify it. If you do it, the lattice will be change.
            
        # lattice
        lattice=poscar['lattice']
        for i in range(0, lattice.shape[0]):
            output.write(' %22.16f %22.16f %22.16f\n' %(lattice[i][0], lattice[i][1], lattice[i][2]))
                
        # element
        elements=poscar['elements']
        for i in range(0, elements.shape[0]):
            output.write('%5s' %elements[i])
        output.write('\n')
        # number
        numbers=poscar['numbers']
        for i in range(0, numbers.shape[0]):
            output.write('%5d' %numbers[i])
        output.write('\n')
        
        
        if isContainedConstraints:
            output.write('Selective dynamics\n')
            
            # type
            output.write(poscar['type']+'\n')
            
            # postions
            positions=poscar['positions']
            constraints=poscar['constraints']
            for i in range(0, positions.shape[0]):
                output.write('%16.8f %16.8f %16.8f' %(positions[i][0], positions[i][1], positions[i][2]))
                constraint=['T' if s0 else 'F' for s0 in constraints[i]]
                output.write('%4s %4s %4s\n' %(constraint[0], constraint[1], constraint[2]))
        else:
            # type
            output.write(poscar['type']+'\n')
            
            # postions
            positions=poscar['positions']
            #constraints=poscar['constraints']
            for i in range(0, positions.shape[0]):
                output.write('%16.8f %16.8f %16.8f\n' %(positions[i][0], positions[i][1], positions[i][2]))
             
        if isContainedVelocities:
            output.write('\n')
            velocities=poscar['velocities']
            for i in range(0, velocities.shape[0]):
                output.write('  %16.8f %16.8f %16.8f\n' %(velocities[i][0], velocities[i][1], velocities[i][2]))
                    
        output.close()    
    
    def __writeXYZ(self):
        pass
    
    def __writeMOL(self):
        pass
        
        
        
        
        
