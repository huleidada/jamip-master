# basic parameters and defaults setting % 
_CONTROL={\
'calculation':None,
'title':None,
'verbosity':None,
'restart_mode':None,
'wf_collect':None,
'nstep':None,
'iprint':None,
'tstress':None,
'tprnfor':None,
'dt':None,
'outdir':None,
'wfcdir':None,
'prefix':None,
'lkpoint_dir':None,
'max_seconds':None,
'etot_conv_thr':None,
'forc_conv_thr':None,
'disk_io':None,
'pseudo_dir':None,
'tefield':None,
'dipfield':None,
'lelfield':None,
'nberrycyc':None,
'lorbm':None,
'lberry':None,
'gdir':None,
'nppstr':None,
'lfcpopt':None,
'gate':None
}

_SYSTEM = {\
'ibrav':None,
'celldm':None,
'A':None,
'B':None,
'C':None,
'cosAB':None,
'cosAC':None,
'cosBC':None,
'nat':None,
'ntyp':None,
'nbnd':None,
'tot_charge':None,
'starting_charge':None,
'tot_magnetization':None,
'starting_magnetization':None,
'ecutwfc':None,
'ecutrho':None,
'ecutfock':None,
'nr1':None,
'nr2':None,
'nr3':None,
'nr1s':None,
'nr2s':None,
'nr3s':None,
'nosym':None,
'nosym_evc':None,
'noinv':None,
'no_t_rev':None,
'force_symmorphic':None,
'use_all_frac':None,
'occupations':None,
'one_atom_occupations':None,
'starting_spin_angle':None,
'degauss':None,
'smearing':None,
'nspin':None,
'noncolin':None,
'ecfixed':None,
'qcutz':None,
'q2sigma':None,
'input_dft':None,
'ace':None,
'exx_fraction':None,
'screening_parameter':None,
'exxdiv_treatment':None,
'x_gamma_extrapolation':None,
'ecutvcut':None,
'nqx1':None,
'nqx2':None,
'nqx3':None,
'localization_thr':None,
'lda_plus_u':None,
'lda_plus_u_kind':None,
'Hubbard_U':None,
'Hubbard_J0':None,
'Hubbard_V':None,
'Hubbard_alpha':None,
'Hubbard_beta':None,
'Hubbard_J':None,
'starting_ns_eigenvalue':None,
'U_projection_type':None,
'Hubbard_parameters':None,
'ensemble_energies':None,
'edir':None,
'emaxpos':None,
'eopreg':None,
'eamp':None,
'angle1':None,
'angle2':None,
'lforcet':None,
'constrained_magnetization':None,
'fixed_magnetization':None,
'lambda':None,
'report':None,
'lspinorb':None,
'assume_isolated':None,
'esm_bc':None,
'esm_w':None,
'esm_efield':None,
'esm_nfit':None,
'fcp_mu':None,
'vdw_corr':None,
'london':None,
'london_s6':None,
'london_c6':None,
'london_rvdw':None,
'london_rcut':None,
'dftd3_version':None,
'dftd3_threebody':None,
'ts_vdw_econv_thr':None,
'ts_vdw_isolated':None,
'xdm':None,
'xdm_a1':None,
'xdm_a2':None,
'space_group':None,
'uniqueb':None,
'origin_choice':None,
'rhombohedral':None,
'zgate':None,
'relaxz':None,
'block':None,
'block_1':None,
'block_2':None,
'block_height':None
}

_ELECTRONS={\
'electron_maxstep':None,
'scf_must_converge':None,
'conv_thr':None,
'adaptive_thr':None,
'conv_thr_init':None,
'conv_thr_multi':None,
'mixing_mode':None,
'mixing_beta':None,
'mixing_ndim':None,
'mixing_fixed_ns':None,
'diagonalization':None,
'diago_thr_init':None,
'diago_cg_maxiter':None,
'diago_david_ndim':None,
'diago_full_acc':None,
'efield':None,
'efield_cart':None,
'efield_phase':None,
'startingpot':None,
'startingwfc':None,
'tqr':None,
'real_space':None,
}

_IONS={\
'ion_positions':None,
'ion_velocities':None,
'ion_dynamics':None,
'pot_extrapolation':None,
'wfc_extrapolation':None,
'remove_rigid_rot':None,
'ion_temperature':None,
'tempw':None,
'tolp':None,
'delta_t':None,
'nraise':None,
'refold_pos':None,
'upscale':None,
'bfgs_ndim':None,
'trust_radius_max':None,
'trust_radius_min':None,
'trust_radius_ini':None,
'w_1':None,
'w_2':None,
}

_CELL ={\
'cell_dynamics':None,
'press':None,
'wmass':None,
'cell_factor':None,
'press_conv_thr':None,
'cell_dofree':None,
}

_PHONON ={\
'amass':None,
'outdir':None,
'prefix':None,
'niter_ph':None,
'tr2_ph':None,
'alpha_mix(niter)':None,
'nmix_ph':None,
'verbosity':None,
'reduce_io':None,
'max_seconds':None,
'fildyn':None,
'fildrho':None,
'fildvscf':None,
'epsil':None,
'lrpa':None,
'lnoloc':None,
'trans':None,
'lraman':None,
'eth_rps':None,
'eth_ns':None,
'dek':None,
'recover':None,
'low_directory_check':None,
'only_init':None,
'qplot':None,
'q2d':None,
'q_in_band_form':None,
'electron_phonon':None,
'el_ph_nsigma':None,
'el_ph_sigma':None,
'ahc_dir':None,
'ahc_nbnd':None,
'ahc_nbndskip':None,
'skip_upperfan':None, 
'lshift_q':None,
'zeu':None,
'zue':None,
'elop':None,
'fpol':None,
'ldisp':None,
'nogg':None,
'asr':None,
'ldiag':None,
'lqdir':None,
'search_sym':None,
'nq1':None,
'nq2':None,
'nq3':None,
'nk1':None,
'nk2':None,
'nk3':None,
'k1':None,
'k2':None,
'k3':None,
'diagonalization':None,
'read_dns_bare':None,
'ldvscf_interpolate':None,
'wpot_dir':None,
'do_long_range':None,
'do_charge_neutral':None,
'start_irr':None,
'last_irr':None,
'nat_todo':None,
'modenum':None,
'start_q':None,
'last_q':None,
'dvscf_star':None,
'drho_star':None,
}

def pwscf_format(params):

    control = {}
    system = {}
    electrons = {}
    ions = {}
    cell = {}
    for key,value in params.items():

        # value %
        if value == None:
            print("Error: The value of the key '%s' \
                            shouldn't be None" % key)
            continue
        elif value == '':
            print("Error: The key '%s' should have \
                                    a value." % key)
            continue
        elif isinstance(value,float):
            value = ('%e' %params[key]).replace('e','d')
        elif key == 'calculation':
             pass
        elif isinstance(value,str):
            value = "'%s'" %value

        # key %
        if key in _CONTROL:
            control[key] = value
        elif key in _SYSTEM:
            system[key] = value
        elif key in _ELECTRONS:
            electrons[key] = value
        elif key in _IONS:
            ions[key] = value
        elif key in _CELL:
            cell[key] = value
        else:
            print("Invalid key %s in pwscf.in ." %key)

    return control, system, electrons, ions, cell

def phonon_format(params):

    result = {}
    for key,value in params.items():

        # key %
        if key in _PHONON:
            result[key] = value

    return result
