from typing import NamedTuple
from functools import lru_cache
import numpy as np
import pandas as pd
from importlib.resources import files

datafile = files(__package__).joinpath("elementInfo.csv")
ElementData = pd.read_csv(datafile,float_precision='round_trip').set_index('symbol', drop=False)
ElementDict = ElementData.to_dict('index')
datafile = files(__package__).joinpath('shannon_rad.csv')
IonRadiusData = pd.read_csv(datafile)

class Element(NamedTuple):
    symbol: str
    Z: int
    name: str
    row: int
    col: int
    mass: float
    atomic_radius: float
    X: float
    covalent_radius: float
    electron_affinity: float
    first_ionization_energy: float

    @property
    def electronegativity(self):
        return self.X

    def __str__(self):
        return 'ELement {}'.format(self.symbol)    

    @classmethod
    @lru_cache(maxsize=128)
    def from_symbol(cls, symbol:str):
        return cls(**ElementDict[symbol])

    @classmethod
    @lru_cache(maxsize=128)
    def from_number(cls, number:int):
        from atomic_number import atomic
        symbol = atomic[number]
        return cls(**ElementDict[symbol])

class Memoize:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@Memoize
def get_element_from_symbol(symbol):
    return Element.from_symbol(symbol)

def get_shannon_radius(symbol, valence:int=None, coord:int=6):
    """ Get shannon atomic radius from csv file
    #Rec,ION,OX_State,Elec_Config.,Coord,Spin_State,Crystal_Radius,Ionic_Radius,NOTES,Z/IR,ele

    Args:
        specie (str): element symbol
        valence (int): element valence
        coord (int): element coordination number

    Returns:
        float: shannon atomic radius
    """
    # if valence not exists, use average instead. 
    sdf = IonRadiusData[(IonRadiusData['ele']==symbol)]
    if len(sdf) == 0: 
        print(symbol, "not found in shannon_rad.csv")
        raise
    # filter valence
    if valence is None:
        tmp = sdf[(sdf['OX_State']==valence)]
        if len(tmp) > 0:
            sdf = tmp
    # filter coordination
    if coord is None:
        tmp = sdf[sdf['Coord']==coord]
        if len(tmp) > 0:
            sdf = tmp

    return sdf['Ionic_Radius'].mean()
    
if __name__ == "__main__":
    import time
    from mendeleev import element

    #print(ElementData['radius'].values)
    data = []
    for i in ElementData['symbol']:
        try:
            elm = element(i)
            r1 = elm.covalent_radius_cordero
            #data.append(elm.covalent_radius_cordero)
        except:
            #print(elm)
            r1 = None
        data.append(r1)
    
    df = pd.read_csv(datafile,float_precision='round_trip')
    df['covalent_radius'] = data
    df.to_csv('cr.csv', index=0)

    '''
    '_sa_instance_state', 'boiling_point', 'discovery_location', 'is_radioactive', 'cas', 'discovery_year', 'jmol_color', 
    'covalent_radius_bragg', 'electron_affinity', 'lattice_constant', 'covalent_radius_cordero', 'en_allen', 'lattice_structure', 
    'covalent_radius_pyykko', 'en_ghosh', 'melting_point', 'covalent_radius_pyykko_double', 'en_pauling', 'mendeleev_number', 
    'covalent_radius_pyykko_triple', 'evaporation_heat', 'metallic_radius', 'c6', 'fusion_heat', 'metallic_radius_c12', 'c6_gb', 
    'gas_basicity', 'molcas_gv_color', 'cpk_color', 'geochemical_class', 'name', 'density', 'glawe_number', 'name_origin', 
    'description', 'goldschmidt_class', 'period', 'dipole_polarizability', 'group_id', 'pettifor_number', 
    'dipole_polarizability_unc', 'heat_of_formation', 'proton_affinity', 'discoverers', 'is_monoisotopic', 'sources', 
    'specific_heat', 'symbol', 'thermal_conductivity', 'econf', 'uses', '_series_id', 'vdw_radius', 'vdw_radius_alvarez', 
    'abundance_crust', 'vdw_radius_bondi', 'abundance_sea', 'annotation', 'vdw_radius_truhlar', 'atomic_number', 'vdw_radius_rt', 
    'atomic_radius', 'vdw_radius_batsanov', 'atomic_radius_rahm', 'vdw_radius_dreiding', 'atomic_volume', 'vdw_radius_uff', 
    'vdw_radius_mm3', 'atomic_weight', 'atomic_weight_uncertainty', 'block', 'group', '_series', 'ionic_radii', 
    '_ionization_energies', '_oxidation_states', 'isotopes', 'screening_constants', 'ec'
    '''
