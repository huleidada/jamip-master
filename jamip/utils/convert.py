import numpy as np

def format_bool(value, default=True):
    if value == True or value == False:
        return value
    elif isinstance(value, str):
        if value[0] == 'T' or value[0] == 't':
            return True
        elif value[0] == 'F' or value[0] == 'f':
            return False
    return default

latex = ['alpha','beta','gamma','delta','epsilon','varepsilon','zeta','eta',
         'theta','vartheta','iota','kappa','lambda','mu','nu','xi','pi','varpi',
         'rho','varrho','sigma','varsigma','tau','upsilon','phi','varphi',
         'chi','psi','omega']

def format_latex(string):
    import re
    result = re.match(r'[A-Za-z]+',string)
    # A -> A
    if len(string) <= 1:
        restring = string
    # \\gamma -> $\gamma$
    elif '\\' in string:
        restring = f'${string}$'
    # Gamma -> $\Gamma$ ; Gamma0 -> $\Gamma_0$ 
    elif result != None and result.group().lower() in latex:
        restring = f'$\\{result.group().capitalize()}$'
        if result.end() < len(string):
            suffix = string[result.end():]
            print(result.group(),result.end(),suffix, string[result.end()])
            if string[result.end()] != '_':
                restring = f'{restring}$_{suffix}$'
            else:
                restring = f'{restring}${suffix}$'
    else:     
        restring = f'{string[0]}$_{string[1:]}$'
        
    return restring
           
def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def counter(obj):

    value, numbers = np.unique(obj, return_counts=True) 
    data = dict(zip(value.tolist(), numbers.tolist()))
    return data

def kpath2list(kpath):
    klabels = []

    for index,kpts in enumerate(kpath):
        if index == 0:                     # start 
            label = format_latex(kpts[0])
            klabels.append(label)
        for kpt in kpts[1:-1]:             # inside
            label = format_latex(kpt)
            klabels.append(label)
        if index < len(kpath)-1:           # edge
            label = format_latex(kpts[-1]) + '|' + format_latex(kpath[index+1][0])
            klabels.append(label)
        else:                              # end
            label = format_latex(kpts[-1])
            klabels.append(label)

    return klabels 

def spectral_function(energy, weight=None, num=601, sigma=0.02, e0=None, smear_type='Lorentz', kpoint_weight=None):
    '''
    https://github.com/QijingZheng/VaspBandUnfolding

    Generate the spectral function

        A(k_i, E) = Σ_m P_{Km}(k_i)Δ(E - Em)

    Where the Δ function can be approximated by Lorentzian or Gaussian function.
    '''
    from scipy.interpolate import interp1d

    # for PROCAR: x=bands, y=procar
    def LorentzSmearing(x, x0, sigma):
        # smearing from x0 to x
        return 1. / np.pi * sigma**2 / ((x - x0)**2 + sigma**2)

    def GaussianSmearing(x, x0, sigma):
        norm_factor = 1 / (sigma * np.sqrt(2 * np.pi))
        return np.exp(-(x - x0)**2 / (2*sigma**2)) / norm_factor

    def GaussianInterpSmearing(x, x0, sigma):
        grid = np.linspace(-6*sigma, 6*sigma, 300)
        gauss = GaussianSmearing(grid, 0, sigma)
        gauss_interp = interp1d(grid, gauss, bounds_error=False, fill_value=0)
        #gauss_interp = interp1d(grid, gauss, fill_value=0)
        return gauss_interp(x - x0)

    # spectral function
    #print(energy.shape, weight.shape)
    #assert energy.shape == weight.shape, weight
    nspin = energy.shape[0]
    nkpts = energy.shape[1]
    SF = np.zeros((nspin, num, nkpts), dtype=float)
    if e0 is None:
        e0 = np.linspace(energy.min() - 5 * sigma, 
                         energy.max() + 5 * sigma, num)

    if smear_type.lower() == 'lorentz':
        func = LorentzSmearing
    elif smear_type.lower() == 'gaussian':
        func = GaussianSmearing
    elif smear_type.lower() == 'gaussianinterp':
        func = GaussianInterpSmearing
    else:
        raise KeyError("unknown smear_type %s" %smear_type)

    print(func)
    for ispin in range(nspin):
        for ii in range(nkpts):
            E_Km = energy[ispin, ii, :]
            if weight is None:
                P_Km = np.ones_like(E_Km)
            else:
                P_Km = weight[ispin, ii, :]
            if kpoint_weight is None:
                kw = 1
            else:
                kw = kpoint_weight[ii]

            SF[ispin, :, ii] = np.sum(
                func(
                    e0[:, None], E_Km[None, :],
                    sigma=sigma
                ) * P_Km[None, :], axis=1
            ) * kw

    return e0, SF
