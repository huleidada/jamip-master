# basic parameters and defaults setting % 
_GLOBAL={\
"DBCSR":None,
"FM":None,
"FM_DIAG_SETTINGS":None,
"GRID":None,
"PRINT":None,
"PRINT_ELPA":None,
"PROGRAM_RUN_INFO":None,
"REFERENCES":None,
"TIMINGS":None,
"ALLTOALL_SGL":None,
"BLACS_GRID":None,
"BLACS_REPEATABLE":None,
"CALLGRAPH":None,
"CALLGRAPH_FILE_NAME":None,
"DLAF_NEIGVEC_MIN":None,
"ECHO_ALL_HOSTS":None,
"ECHO_INPUT":None,
"ELPA_KERNEL":None,
"ELPA_NEIGVEC_MIN":None,
"ELPA_QR":None,
"ELPA_QR_UNSAFE":None,
"ENABLE_MPI_IO":None,
"EPS_CHECK_DIAG":None,
"EXTENDED_FFT_LENGTHS":None,
"FFTW_PLAN_TYPE":None,
"FFTW_WISDOM_FILE_NAME":None,
"FFT_POOL_SCRATCH_LIMIT":None,
"FLUSH_SHOULD_FLUSH":None,
"OUTPUT_FILE_NAME":None,
"PREFERRED_DGEMM_LIBRARY":None,
"PREFERRED_DIAG_LIBRARY":None,
"PREFERRED_FFT_LIBRARY":None,
"PRINT_LEVEL":None,
"PROGRAM_NAME":None,
"PROJECT_NAME":None,
"PROJECT":None,
"RUN_TYPE":None,
"SAVE_MEM":None,
"SEED":None,
"TRACE":None,
"TRACE_MASTER":None,
"TRACE_MAX":None,
"TRACE_ROUTINES":None,
"WALLTIME":None,
}

_FORCE_EVAL = {\
"BSSE":None,
"DFT":None,
"EIP":None,
"EMBED":None,
"EXTERNAL_POTENTIAL":None,
"MIXED":None,
"MM":None,
"NNP":None,
"PRINT":None,
"PROPERTIES":None,
"PW_DFT":None,
"QMMM":None,
"RESCALE_FORCES":None,
"SUBSYS":None,
"METHOD":None,
"STRESS_TENSOR":None,
}

_DFT={\
"ACTIVE_SPACE":None,
"ALMO_SCF":None,
"AUXILIARY_DENSITY_MATRIX_METHOD":None,
"DENSITY_FITTING":None,
"EFIELD":None,
"ENERGY_CORRECTION":None,
"EXCITED_STATES":None,
"EXTERNAL_DENSITY":None,
"EXTERNAL_POTENTIAL":None,
"EXTERNAL_VXC":None,
"HARRIS_METHOD":None,
"KG_METHOD":None,
"KPOINTS":None,
"LOCALIZE":None,
"LOW_SPIN_ROKS":None,
"LS_SCF":None,
"MGRID":None,
"PERIODIC_EFIELD":None,
"POISSON":None,
"PRINT":None,
"QS":None,
"REAL_TIME_PROPAGATION":None,
"RELATIVISTIC":None,
"SCCS":None,
"SCF":None,
"SCRF":None,
"SIC":None,
"SMEAGOL":None,
"TDDFPT":None,
"TRANSPORT":None,
"XAS":None,
"XAS_TDP":None,
"XC":None,
"AUTO_BASIS":None,
"BASIS_SET_FILE_NAME":None,
"CHARGE":None,
"CORE_CORR_DIP":None,
"EXCITATIONS":None,
"MULTIPLICITY":None,
"PLUS_U_METHOD":None,
"POTENTIAL_FILE_NAME":None,
"RELAX_MULTIPLICITY":None,
"ROKS":None,
"SORT_BASIS":None,
"SUBCELLS":None,
"SURFACE_DIPOLE_CORRECTION":None,
"SURF_DIP_DIR":None,
"SURF_DIP_POS":None,
"SURF_DIP_SWITCH":None,
"UKS":None,
"WFN_RESTART_FILE_NAME":None,
}

_CELL_OPT={\
"BFGS":None,
"CG":None,
"LBFGS":None,
"PRINT":None,
"CONSTRAINT":None,
"EPS_SYMMETRY":None,
"EXTERNAL_PRESSURE":None,
"KEEP_ANGLES":None,
"KEEP_SPACE_GROUP":None,
"KEEP_SYMMETRY":None,
"MAX_DR":None,
"MAX_FORCE":None,
"MAX_ITER":None,
"OPTIMIZER":None,
"PRESSURE_TOLERANCE":None,
"RMS_DR":None,
"RMS_FORCE":None,
"SPGR_PRINT_ATOMS":None,
"STEP_START_VAL":None,
"SYMM_EXCLUDE_RANGE":None,
"SYMM_REDUCTION":None,
"TYPE":None,
}

_MD={\
"ADIABATIC_DYNAMICS":None,
"AVERAGES":None,
"BAROSTAT":None,
"CASCADE":None,
"INITIAL_VIBRATION":None,
"LANGEVIN":None,
"MSST":None,
"PRINT":None,
"REFTRAJ":None,
"RESPA":None,
"SHELL":None,
"THERMAL_REGION":None,
"THERMOSTAT":None,
"VELOCITY_SOFTENING":None,
"ANGVEL_TOL":None,
"ANGVEL_ZERO":None,
"ANNEALING":None,
"ANNEALING_CELL":None,
"COMVEL_TOL":None,
"DISPLACEMENT_TOL":None,
"ECONS_START_VAL":None,
"ENSEMBLE":None,
"INITIALIZATION_METHOD":None,
"MAX_STEPS":None,
"SCALE_TEMP_KIND":None,
"STEPS":None,
"STEP_START_VAL":None,
"TEMPERATURE":None,
"TEMPERATURE_ANNEALING":None,
"TEMP_KIND":None,
"TEMP_TOL":None,
"TIMESTEP":None,
"TIME_START_VAL":None,
}

_GEO_OPT={\
"BFGS":None,
"CG":None,
"LBFGS":None,
"PRINT":None,
"TRANSITION_STATE":None,
"EPS_SYMMETRY":None,
"KEEP_SPACE_GROUP":None,
"MAX_DR":None,
"MAX_FORCE":None,
"MAX_ITER":None,
"OPTIMIZER":None,
"RMS_DR":None,
"RMS_FORCE":None,
"SPGR_PRINT_ATOMS":None,
"STEP_START_VAL":None,
"SYMM_EXCLUDE_RANGE":None,
"SYMM_REDUCTION":None,
"TYPE":None,
}

_CONSTRAINT={\
"COLLECTIVE":None,
"COLVAR_RESTART":None,
"CONSTRAINT_INFO":None,
"FIXED_ATOMS":None,
"FIX_ATOM_RESTART":None,
"G3X3":None,
"G4X6":None,
"HBONDS":None,
"LAGRANGE_MULTIPLIERS":None,
"VIRTUAL_SITE":None,
"CONSTRAINT_INIT":None,
"PIMD_BEADWISE_CONSTRAINT":None,
"ROLL_TOLERANCE":None,
"SHAKE_TOLERANCE":None,
}

_BAND={\
"BANNER":None,
"CI_NEB":None,
"CONVERGENCE_CONTROL":None,
"CONVERGENCE_INFO":None,
"ENERGY":None,
"OPTIMIZE_BAND":None,
"PROGRAM_RUN_INFO":None,
"REPLICA":None,
"REPLICA_INFO":None,
"STRING_METHOD":None,
"ALIGN_FRAMES":None,
"BAND_TYPE":None,
"K_SPRING":None,
"NPROC_REP":None,
"NUMBER_OF_REPLICA":None,
"POT_TYPE":None,
"PROC_DIST_TYPE":None,
"ROTATE_FRAMES":None,
"USE_COLVARS":None,
}

_MOTION={\
"BAND":None,
"CELL_OPT":None,
"CONSTRAINT":None,
"DRIVER":None,
"FLEXIBLE_PARTITIONING":None,
"FREE_ENERGY":None,
"GEO_OPT":None,
"MC":None,
"MD":None,
"PINT":None,
"PRINT":None,
"SHELL_OPT":None,
"TMC":None,
}

def cp2k_format(incar):
    from copy import deepcopy  
    from collections import defaultdict

    config = defaultdict(dict)
    # initialize depth2 & sort     
    config['GLOBAL'] = {}
    config['FORCE_EVAL']['DFT'] = defaultdict(dict)

    # GET DFT
    for key,value in incar.items():

        key = key.upper()

        if value == None:
            print(f"Error: The value of the key '{key}' shouldn't be None")
            continue
        elif value == '':
            print(f"Error: The key '{key}' should have a value.")
            continue
        elif isinstance(value,str):
            value = f"{value.upper()}" 

        # key %
        if key in _GLOBAL:
            config['GLOBAL'][key] = value

        elif key in _FORCE_EVAL:
            config['FORCE_EVAL'][key] = value

        elif key in _MOTION:
            config['MOTION'][key] = value

        elif key in _DFT:
            config['FORCE_EVAL']['DFT'][key] = value

        elif isinstance(value, dict):
             set_value(value, config[key])

        #else:
        #    print("Invalid key %s in pwscf.in ." %key)

    return config

def set_value(source, output):
    for key,value in source.items():
        ukey = key.upper()
        if isinstance(value, dict) and ukey in output:
            set_value(source[key], output[ukey])
        else:
            #print('u',ukey, type(value), key in output, output.keys())
            output[ukey] = source[key] 
