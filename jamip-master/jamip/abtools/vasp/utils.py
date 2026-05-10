# Recommended potentials for DFT calculations

paw_potcar =  {'H': 'H',  'He': 'He',
              'Li': 'Li_sv', 'Be': 'Be', 'B' : 'B',  'C' : 'C',  'N' : 'N', 'O' : 'O', 'F' : 'F',  'Ne': 'Ne', 
              'Na': 'Na_pv', 'Mg': 'Mg', 'Al': 'Al', 'Si': 'Si', 'P' : 'P', 'S' : 'S', 'Cl': 'Cl', 'Ar': 'Ar', 
               'K': 'K_sv',  'Ca': 'Ca_sv', 'Sc': 'Sc_sv', 'Ti': 'Ti_sv', 'V' : 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv', 'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 
              'Cu': 'Cu', 'Zn': 'Zn', 'Ga': 'Ga_d', 'Ge': 'Ge_d', 'As': 'As', 'Se': 'Se', 'Br': 'Br', 'Kr': 'Kr', 
              'Rb': 'Rb_sv', 'Sr': 'Sr_sv', 'Y' : 'Y_sv', 'Zr': 'Zr_sv', 'Nb': 'Nb_sv', 'Mo': 'Mo_sv', 'Tc': 'Tc_pv', 'Ru': 'Ru_pv', 'Rh': 'Rh_pv', 'Pd': 'Pd', 
              'Ag': 'Ag', 'Cd': 'Cd', 'In': 'In_d', 'Sn': 'Sn_d', 'Sb': 'Sb', 'Te': 'Te', 'I' : 'I', 'Xe': 'Xe', 
              'Cs': 'Cs_sv', 'Ba': 'Ba_sv', 'Lu': 'Lu_3', 'Hf': 'Hf_pv', 'Ta': 'Ta_pv', 'W' : 'W_sv', 'Re': 'Re', 'Os': 'Os', 'Ir': 'Ir', 'Pt': 'Pt', 
              'Au': 'Au', 'Hg': 'Hg', 'Tl': 'Tl_d', 'Pb': 'Pb_d', 'Bi': 'Bi_d', 'Po': 'Po_d', 'At': 'At', 'Rn': 'Rn', 
              'Fr': 'Fr_sv', 'Ra': 'Ra_sv', 
    
              'La': 'La', 'Ce': 'Ce', 'Pr': 'Pr_3', 'Nd': 'Nd_3', 'Pm': 'Pm_3', 'Sm': 'Sm_3', 'Eu': 'Eu_2', 'Gd': 'Gd_3', 'Tb': 'Tb_3', 'Dy': 'Dy_3', 
              'Ho': 'Ho_3', 'Er': 'Er_3', 'Tm': 'Tm_3', 'Yb': 'Yb_2', 'Ac': 'Ac', 'Th': 'Th', 'Pa': 'Pa', 'U' : 'U', 'Np': 'Np', 'Pu': 'Pu', 
              'Am': 'Am', 'Cm': 'Cm'}

gw_potcar =  {'H': 'H_GW',  'He': 'He_GW',
             'Li': 'Li_sv_GW', 'Be': 'Be_sv_GW', 'B' : 'B_GW',  'C' : 'C_GW',  'N' : 'N_GW', 'O' : 'O_GW', 'F' : 'F_GW',  'Ne': 'Ne_GW', 
             'Na': 'Na_sv_GW', 'Mg': 'Mg_sv_GW', 'Al': 'Al_GW', 'Si': 'Si_GW', 'P' : 'P_GW', 'S' : 'S_GW', 'Cl': 'Cl_GW', 'Ar': 'Ar_GW', 
              'K': 'K_sv_GW',  'Ca': 'Ga_sv_GW', 'Sc': 'Sc_sv_GW', 'Ti': 'Ti_sv_GW', 'V' : 'V_sv_GW', 'Cr': 'Cr_sv_GW', 'Mn': 'Mn_sv_GW', 'Fe': 'Fe_sv_GW', 'Co': 'Co_sv_GW', 
             'Ni': 'Ni_sv_GW', 'Cu': 'Cu_sv_GW', 'Zn': 'Zn_sv_GW', 'Ga': 'Ga_d_GW',  'Ge': 'Ge_d_GW', 'As': 'As_GW', 'Se': 'Se_GW', 'Br': 'Br_GW', 'Kr': 'Kr_GW', 
             'Rb': 'Rb_sv_GW', 'Sr': 'Sr_sv_GW', 'Y' : 'Y_sv_GW',  'Zr': 'Zr_sv_GW', 'Nb': 'Nb_sv_GW', 'Mo': 'Mo_sv_GW', 'Tc': 'Tc_sv_GW', 'Ru': 'Ru_sv_GW', 'Rh': 'Rh_sv_GW', 
             'Pd': 'Pd_sv_GW', 'Ag': 'Ag_sv_GW', 'Cd': 'Cd_sv_GW', 'In': 'In_d_GW',  'Sn': 'Sn_d_GW', 'Sb': 'Sb_d_GW', 'Te': 'Te_GW', 'I' : 'I_GW', 'Xe': 'Xe_GW', 
             'Cs': 'Cs_sv_GW', 'Ba': 'Ba_sv_GW', 'La': 'La_GW', 'Ce': 'Ce_GW', 'Hf': 'Hf_sv_GW', 'Ta': 'Ta_sv_GW', 'W' : 'W_sv_GW', 'Re': 'Re_sv_GW', 'Os': 'Os_sv_GW', 
             'Ir': 'Ir_sv_GW', 'Pt': 'Pt_sv_GW', 'Au': 'Au_sv_GW', 'Hg': 'Hg_sv_GW', 'Tl': 'Tl_d_GW', 'Pb': 'Pb_d_GW', 'Bi': 'Bi_d_GW', 'Po': 'Po_d_GW', 'At': 'At_d_GW', 'Rn': 'Rn_d_GW', 
             }

# basic parameters and defaults setting % 

__incar__={'dimension':{\
'NKPTS':None,
'NKDIM':None,
'NBANDS':None,
'NEDOS': None,
'NIONS': None,
'NPLWV': None,
'NGX': None,
'NGY': None,
'NGZ': None,
'NGXF':None,
'NGYF':None,
'NGZF':None, 
'SYSTEM':'JAMIP'},
'starts':{\
'PREC':'Normal',
'ISTART':0, 
'ICHARG':2,
'ISPIN':1,
'LNONCOLLINEAR':None, 
'LSORBIT':False,
'INIWAV':None,
'LASPH':None, 
'METAGGA':None},
'electronic':{\
'ENCUT':1.3,
'ENINI':None,
'NELM': None,
'EDIFF':1E-5,
'LREAL': 'Auto',
'NLSPLINE':None, 
'LCOMPAT':None,
'GGA_COMPAT':None,
'LMAXPAW': None,
'LMAXMIX':None,
'VOSKOWN':None,
'IALGO': None,
'LDIAG': None,
'LSUBROT':None, 
'TURBO':None,
'IRESTART':None, 
'NREBOOT':None, 
'NMIN':None,
'EREF': None, 
'IMIX':None, 
'AMIX':None, 
'BMIX':None, 
'AMIX_MAG':None,
'BMIX_MAG':None,  
'AMIN':None, 
'WC':None, 
'INIMIX':None,
'MIXPRE':None,
'MAXMIX':None,
 'ALGO': 'Fast'},
'ionic':{\
'EDIFFG':1E-4,
'NSW':0,
'NBLOCK':None,
'IBRION':None,
'NFREE':None,
'ISIF':2,
'IWAVPR':None, 
'ISYM':2,
'LCORR':None,
'POTIM':None,
'TEIN': None,
'TEBEG':None,
'SMASS':None,
'SCALEE':None,
'NPACO':None, 
'PSTRESS':None, 
'NELECT':None,
'NUPDOWN':None},
'dos':{\
'EMIN':None,
'ENMAX':None,
'EFERMI':None,
'ISMEAR':0,
'SIGMA':0.2},
'intra_band':{\
'WEIMIN':None, 
'EBREAK':None,
'DEPER':None, 
'TIME':None},
'write_flags':{\
'LWAVE':False,
'LDOWNSAMPLE':None,
'LCHARG':False,
'LVTOT':None,
'LVHAR':None, 
'LELF':None, 
'LORBIT':0},
'dipole':{\
'LMONO':None,
'LDIPOL':None,
'IDIPOL':None,
'EPSILON':None},
'XC_func':{\
'GGA':None,
'LEXCH':None, 
'VOSKOWN':None,
'LHFCALC':None,
'LHFONE':None, 
'AEXX':None},
'linear':{\
'LEPSILON':None, 
'LRPA':None, 
'LNABLA':None, 
'LVEL':None,
'LINTERFAST':None,
'KINTER':None, 
'CSHIFT':None, 
'OMEGAMAX':None,
'DEG_THRESHOLD':None,
'RTIME':None, 
'WPLASMAI':None},
'orbital_mag':{\
'ORBITALMAG':None, 
'LCHIMAG':None,
'DQ':None, 
'LLRAUG':None},
'parrellel':{\
'NCORE':None,
'NPAR':None,
'KPAR':None,
'NSIM':1}}


class KeyWords(object):

  def __init__(self, **kwargs):
       params = __incar__
       params.update(kwargs)
       self.__set_keywords__(params)


  def __set_keywords__(self, params=None):
       for key in params:
           if key not in self.__dict__:
               is_set = False
               for k in params[key]: 
                   if params[key][k] is not None: 
                       is_set = True
               if is_set: 
                   self.__dict__[key] = self.__parameters__(params[key])
  
  def __parameters__(self, name):
	
        class Parameters(object):
            pass 
        p = Parameters()
        for key in name:
            if key not in p.__dict__ and name[key] is not None:
                p.__dict__[key] = name[key]
	
        return p


def dict_get(dict, key, default):
    import types
    tmp = dict
    for k,v in tmp.items():
        if k == key:
            return v
        else:
            if type(v) is types.DictType:
                ret = dict_get(v, key, default)
                if ret is not default:
                    return ret
    return default

   

# %%%%%%%%%%% appendix code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__default_incar__ ="""\
SYSTEM = {name}

# Start parameter for this run:
    ISTART =  {istart:>9d} ! 0-new  1-cont  2-samecut
    ICHARG =  {lcharge:>9d} ! 1-file 2-atom 10-const
    INIWAV =  {iniwav:>9d} ! 0-lowe 1-rand

# Electronic relaxation :
    ENCUT  =  {encut:>8.2f} 
    IALGO  =  {ialgo:>9d}   
    NELM   =  {nelm:>9d} ! algorithm
    NELMIN =  {nelmin:>9d} ! algorithm
    NELMDL =  {nelmdl:>8d} ! algorithm ELM steps
    EDIFF  =  {ediff:>5.2E} ! stopping-criterion for ELM
    BMIX   =  {bmix:>8.2f}  

# Ionic relaxation :
    EDIFFG =  {eidffg:>5.2E} ! stopping-criterion for IOM
    NSW    =  {nsw:>9d} ! number of steps for IOM
    IBRION =  {ibrion:>9d} ! conjugate gradient for IOM
    POTIM  =  {potim:>8.2f} ! time-step for ion-mtion

# DOS related values :
    SIGMA  =  {sigma:>8.2f} ! broad in eV 
    ISMEAR =  {imear:>9d} ! -4-tet -1-fermi 0-gaus

# Parrellel calculation :
    LPLANE =  {lplane:>9s} ! False-fast network, True-low network  
    NPAR   =  {ncore:>9d} ! sqrt(nodes) 
    NSIM   =  {nsim:>9d} ! 4-low network, 1-fast network

# Orbital info :
    LORBIT =  {lorbit:>9d} ! 10-total, >10-special orbital

# File IO flag :
    LCHARG =  {lcharge:>9s} 	  
    LWAVE  =  {lwave:>9s} 	  
     

     
"""

__database = {\
"NGX": None,
"NGY": None,
"NGZ":		"FFT mesh for orbitals (Sec. 6.3,6.11)",
"NGXF":         "FFT mesh for orbitals (Sec. 6.3,6.11)",
"NGYF":         "FFT mesh for orbitals (Sec. 6.3,6.11)",
"NGZF": 	"FFT mesh for charges (Sec. 6.3,6.11)",
"NBANDS": 	"number of bands included in the calculation (Sec. 6.5)",
"NBLK": 	"blocking for some BLAS calls (Sec. 6.6)",
"SYSTEM": 	"name of System",
"NWRITE": 	"verbosity write-flag (how much is written)",
"ISTART":	"startjob: 0-new 1-cont 2-samecut",
"ICHARG":	"charge: 1-file 2-atom 10-const",
"ISPIN":	"spin polarized calculation (2-yes 1-no)",
"MAGMOM":	"initial mag moment / atom",
"INIWAV":	"initial electr wf. : 0-lowe 1-rand",
"ENCUT":	"energy cutoff in eV",
"PREC":		"VASP.4.5 also: normal, accurate",
"NELM":         " ",
"NELMIN":       " ",
"NELMDL":	"nr. of electronic steps",
"EDIFF":	"stopping-criterion for electronic upd.",
"EDIFFG":	"stopping-criterion for ionic upd.",
"NSW":		"number of steps for ionic upd.",
"NBLOCK":       " ",
"KBLOCK":	"inner block; outer block",
"IBRION":	"ionic relaxation: 0-MD 1-quasi-New 2-CG",
"ISIF":		"calculate stress and what to relax",
"IWAVPR":	"prediction of wf.: 0-non 1-charg 2-wave 3-comb",
"ISYM":		"symmetry: 0-nonsym 1-usesym",
"SYMPREC":	"precession in symmetry routines",
"LCORR":	"Harris-correction to forces",
"POTIM":	"time-step for ion-motion (fs)",
"TEBEG":        " ",
"TEEND":	"temperature during run",
"SMASS":	"Nose mass-parameter (am)",
"NPACO":        " ",
"APACO":	"distance and nr. of slots for P.C.",
"POMASS":	"mass of ions in am",
"ZVAL":		"ionic valence",
"RWIGS":	"Wigner-Seitz radii",
"NELECT":	"total number of electrons",
"NUPDOWN":	"fix spin moment to specified value",
"EMIN":         " ",
"EMAX":		"energy-range for DOSCAR file",
"ISMEAR":       "part. accupaties: -5 Blochl -4-tet -1-fermi 0-gaus > 0 MP",
"SIGMA":	"broadening in eV -4-tet -1-fermi 0-gaus",
"ALGO":		"algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)",
"IALGO":	"algorithm: use only 8 (CG) or 48 (RMM-DIIS)",
"LREAL":	"non-local projectors in real space",
"ROPT":		"number of grid points for non-local proj in real space",
"GGA":		"xc-type: e.g. PE AM or 91",
"VOSKOWN":	"use Vosko, Wilk, Nusair interpolation",
"DIPOL":	"center of cell for dipol",
"AMIX":	        " ",
"BMIX":		"tags for mixing",
"WEIMIN":       " ",
"EBREAK":	"",
"DEPER":	"special control tags",
"TIME":		"special control tag",
"LWAVE":        " ",
"LCHARG":       " ",
"LVTOT":        " ",
"LVHAR":	"create WAVECAR/CHGCAR/LOCPOT",
"LELF":		"create ELFCAR",
"LORBIT":	"create PROOUT",
"NPAR":		"parallelization over bands",
"LSCALAPACK":	"switch off scaLAPACK",
"LSCALU":	"switch of LU decomposition",
"LASYNC":	"overlap communcation with calculations"
}


class LinearPhase(object):

    k = 8.617333262145E10-5 # unit eV/K

    def __init___(self, stdout=None):
        self.stdout = stdout 

    @property
    def free_energy(self,temperature=300):
        """
        omiga: weight;
        temperature: unit K;
        """
        import numpy as np 

        energy = self.energy(temperature)
        x = self.sigma(energy,temperature)
        w = self.weight

	
        R = [] # R = kTln(x_i/w_i)
        for i in range(len(x)):
            R.append(self.k*temperature*np.exp(x[i]/w[i]))
	
        return np.sum(x*(R+energy))
    
    def zeta(self,F,T):
        """
        function: zeta_i 
        F: array, free energy for each configuration;
        T: given temperture;
        """
        import numpy as np 
        return np.exp(-(F/self.k*T))	
   
    @property 
    def weight(self):
        """
        w_i: the multiplicity of the configuration

        # note: this can be solve by other method?
        """
        import numpy as np

        dat = np.loadtxt('multiplicity.dat',dtype=int)	

        dat = dict(zip(dat.T[0],dat.T[1]))
	
        return dat.values() 
