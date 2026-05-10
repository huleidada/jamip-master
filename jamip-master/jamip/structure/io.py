import time

def write_cif(s1,stdout):
    a,b,c,alpha,beta,gamma = s1.lattice_parameters

    with open(stdout,'w') as f:
        f.write('data_create_by_jamip\n')
        f.write('_audit_update_record %s\n' %time.strftime("%Y-%m-%d", time.localtime()))
 
        f.write("_chemical_formula_structural '%s'\n" %s1.get_formula(reduced=True,split=' '))
        f.write("_chemical_formula_sum '%s'\n" %s1.get_formula(split=' '))
        f.write("_chemical_name_structure_type %s\n" %s1.get_formula(reduced=True))
 
        f.write('_cell_length_a     %g\n' % a)
        f.write('_cell_length_b     %g\n' % b)
        f.write('_cell_length_c     %g\n' % c)
        f.write('_cell_angle_alpha     %g\n' % alpha)
        f.write('_cell_angle_beta      %g\n' % beta)
        f.write('_cell_angle_gamma     %g\n' % gamma)
        f.write('_cell_formula_units_Z %s\n' % s1.composition.Z)
        f.write("_symmetry_space_group_name_H-M 'P 1'\n")
        f.write('_symmetry_int_tables_number 1\n')
        f.write('loop_\n')
        f.write('_symmetry_equiv_pos_site_id\n')
        f.write('_symmetry_equiv_pos_as_xyz\n')
        f.write("1 'x, y, z'\n")
        f.write('loop_\n')
        f.write('_atom_site_label\n')
        f.write('_atom_site_type_symbol\n')
        f.write('_atom_site_symmetry_multiplicity\n')
        f.write('_atom_site_Wyckoff_symbol\n')
        f.write('_atom_site_fract_x\n')
        f.write('_atom_site_fract_y\n')
        f.write('_atom_site_fract_z\n')
        f.write('_atom_site_B_iso_or_equiv\n')
        f.write('_atom_site_occupancy\n')
        for e,pos in zip(s1.get_elements(type='symbol'),s1.get_positions(type='direct')):
            f.write('{0} {0} 1 a {1[0]:.5f} {1[1]:.5f} {1[2]:.5f} . 1. \n'.format(e,pos))

def write_pdb(mol, stdout, safe=True):
    """
    REMARK   Materials Studio PDB file
    ATOM      1  F1  UNL C   1       2.441  -1.232  -0.000  1.00  0.00           F
    TER       1
    CONECT    1    4
    END
    """
    from collections import defaultdict
    from .atom import Cell
    from .structure import Structure
    record = defaultdict(int)

    with open(stdout,'w') as f:
        #f.write(f'REMARK  {mol.comment_line}\n')
        f.write(f'REMARK   JAMIP PDB file\n')
        if mol.lattice is not None:
            a,b,c,alpha,beta,gamma = mol.lattice_parameters 
            f.write(f'CRYST1{a:>9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f}\n')
            if safe and isinstance(mol, Structure):
                cell = Cell.from_parameters(a,b,c,alpha,beta,gamma) 
                for atom in mol.atomic_positions:
                    atom.scale_coord = atom.scale_coord
                    atom.lattice = cell
 
        for i,atom in enumerate(mol.atomic_positions):
            record[atom.specie] += 1
            label = f"{atom.specie:>2.2s}{record[atom.specie]:<2d}"
            #label = f"{atom.specie:>2s}  "
            x = atom.coord[0]
            y = atom.coord[1]
            z = atom.coord[2]
            # total length=82
            f.write(f"ATOM  {i+1:>5d} {label:.4s} UNL C   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {atom.specie:>2s}    \n")
            #f.write(f"HATATM{i+1:>5d} {label:<4s} MOL A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {atom.specie:>2s}\n")

        try:
            connects = mol.get_connections(type="pdb")
            f.write(f"TER   {len(mol)+1:>5d}\n")
            for i in range(1,len(mol)+1):
                values = [i] + sorted(list(set(connects[i])))
                line = "".join([f"{j:>5d}" for j in values])
                f.write(f'CONECT{line}\n')
        except:
            connects = []
        f.write("END")

def write_xyz(mol, stdout):

    with open(stdout,'w') as f:
        f.write('%d\n' %len(mol.atomic_positions))
        f.write('Indium dibismuth tetrasulfide chloride\n')

        for atom in mol.atomic_positions:
            f.write('{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n'.format(p=atom.coord, e=atom.specie))

def write_struct(s1, stdout, symprec=1e-5, lattice='P'):
    """
    for wien2k
    """
    import spglib
    import numpy as np

    rot=spglib.get_symmetry_dataset(s1.to_cell(), symprec=symprec)['rotations']
    unrot = np.unique(rot, axis=0)
    # nm -> Bohr
    Bohr = 0.5291772083
    cell = s1.lattice_parameters
    cell[0:3] /= Bohr
    # initialize rmt
    rmt = [2.0] * len(s1)
    
    with open(stdout,'w') as f:
    
        f.write('HTE output'+'\n') # title
        f.write(lattice +
                 '   LATTICE,NONEQUIV.ATOMS:%3i\nMODE OF CALC=RELA\n' % len(s1))

        f.write((' %9.6f' * 6) % tuple(cell) + '\n')
        for idx,atom in enumerate(s1.atomic_positions):
            f.write('ATOM %3i: ' % (idx + 1))
            f.write('X=%10.8f Y=%10.8f Z=%10.8f\n' % tuple(atom.scale_coord))
            f.write('          MULT= 1          ISPLIT= 1\n')
            f.write('%8i: ' % (idx + 1))
            f.write('X=%10.8f Y=%10.8f Z=%10.8f\n' % tuple(atom.scale_coord))
            zz = atom.atomic_number
            if zz > 71:
                ro = 0.000005
            elif zz > 36:
                ro = 0.00001
            elif zz > 18:
                ro = 0.00005
            else:
                ro = 0.0001
            f.write('%-10s NPT=%5i  R0=%9.8f RMT=%10.4f   Z:%10.5f\n' %
                    (atom.specie, 781, ro, rmt[idx], zz))
            f.write('LOCAL ROT MATRIX:    %9.7f %9.7f %9.7f\n' % (1.0, 0.0, 0.0))
            f.write('                     %9.7f %9.7f %9.7f\n' % (0.0, 1.0, 0.0))
            f.write('                     %9.7f %9.7f %9.7f\n' % (0.0, 0.0, 1.0))

        # write symmetry %
        f.write('%4d     NUMBER OF SYMMETRY OPERATIONS\n' %len(unrot))
        for j,mat in enumerate(unrot):
            for i in range(3):
                f.write("%2d %2d %2d 0.0000000\n" %(mat[i,0], mat[i,1], mat[i,2]))
            f.write("%8d\n" %(j+1))

def write_boltztrap(s1, stdout, rotations=None, symprec=1e-5):
    import spglib
    import numpy as np

    if rotations is None:
        sym = spglib.get_symmetry_dataset(s1.to_cell(), symprec=symprec)
        rot = np.unique(sym['rotations'], axis=0)
    else:
        sym = {'number': 2}
        rot = np.array(rotations)

    # nm -> Bohr
    Bohr = 0.5291772083
    lattice = s1.lattice / Bohr

    with open(stdout,'w') as f:
    
        f.write('%s %d\n' %(s1.get_formula(), sym['number'])) # title
        for latt in lattice:
           f.write(("%12.5f" * 3) %tuple(latt) + '\n')

        # write symmetry %
        f.write('%d\n' %len(rot))
        for j,mat in enumerate(rot):
            for i in range(3):
                f.write("%2d %2d %2d\n" %(mat[i,0], mat[i,1], mat[i,2]))
