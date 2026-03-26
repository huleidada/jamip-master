from __future__ import annotations

from ..materials.structure import Structure
from ..materials.composition import Composition
from ..materials.element import Element
from ..materials.species import Species
from ..materials.atom import Atom
from ..materials.entry import Entry
from ..materials.spacegroup import Spacegroup
from ..materials.prototype import Prototype
import re

#import pandas as pd
#import pathlib

class Query(object):

    def __init__(self):
        self.__entry = 'vasp'

    def get_phase(self, **kwargs):
        '''
        chemsys: 'Si-O-Al'
        element_set: ['Si','O']
        '''        

        if 'element_set' in kwargs:
            elements_set = kwargs['element_set']
        elif 'chemsys' in kwargs:
            elements_set = kwargs['chemsys'].split('-')
        elif 'formula' in kwargs:
            elements_set = list(set(re.findall('[A-Z][a-z]?', kwargs['formula'])))
        else:
            raise ValueError("Miss species information.")


    def get_structure_by_id(self, uid, fields=None):
        print(uid)
        query = Structure.objects.filter(name=uid)
        return query

    def get_entry_by_id(self, uid, fields=None):
        print(uid)
        query = Entry.objects.get(name=uid)
        return query

    def summary_search(self,
        band_gap: tuple[float, float] | None = None,
        chemsys: str | list[str] | None = None,
        crystal_system: None = None,
        density: tuple[float, float] | None = None,
        deprecated: bool | None = None,
        e_electronic: tuple[float, float] | None = None,
        e_ionic: tuple[float, float] | None = None,
        e_total: tuple[float, float] | None = None,
        efermi: tuple[float, float] | None = None,
        elastic_anisotropy: tuple[float, float] | None = None,
        elements: list[str] | None = None,
        energy_above_hull: tuple[float, float] | None = None,
        equilibrium_reaction_energy: tuple[float, float] | None = None,
        exclude_elements: list[str] | None = None,
        formation_energy: tuple[float, float] | None = None,
        formula: str | list[str] | None = None,
        g_reuss: tuple[float, float] | None = None,
        g_voigt: tuple[float, float] | None = None,
        g_vrh: tuple[float, float] | None = None,
        has_reconstructed: bool | None = None,
        is_gap_direct: bool | None = None,
        is_metal: bool | None = None,
        is_stable: bool | None = None,
        n: tuple[float, float] | None = None,
        num_elements: tuple[int, int] | None = None,
        num_sites: tuple[int, int] | None = None,
        num_magnetic_sites: tuple[int, int] | None = None,
        num_unique_magnetic_sites: tuple[int, int] | None = None,
        piezoelectric_modulus: tuple[float, float] | None = None,
        poisson_ratio: tuple[float, float] | None = None,
        possible_species: list[str] | None = None,
        shape_factor: tuple[float, float] | None = None,
        spacegroup_number: int | None = None,
        spacegroup_symbol: str | None = None,
        surface_energy_anisotropy: tuple[float, float] | None = None,
        theoretical: bool | None = None,
        total_energy: tuple[float, float] | None = None,
        total_magnetization: tuple[float, float] | None = None,
        total_magnetization_normalized_formula_units: tuple[float, float] | None = None,
        total_magnetization_normalized_vol: tuple[float, float] | None = None,
        uncorrected_energy: tuple[float, float] | None = None,
        volume: tuple[float, float] | None = None,
        weighted_surface_energy: tuple[float, float] | None = None,
        weighted_work_function: tuple[float, float] | None = None,
        sort_fields: list[str] | None = None,
        all_fields: bool = True,
        fields: list[str] | None = None,
        ):

        conditions = {}
        # build conditions
        if formula != None:
           conditions['structure__composition__formula'] = formula
        if band_gap != None:
           conditions['band_gap'] = band_gap
        if num_elements != None:
            i,j = num_elements
            conditions['structure__ntypes__gte'] = i
            conditions['structure__ntypes__lte'] = j

        # build fields
        fieldsets = ['energy_per_atom', 'bandgap']
        if all_fields and not fields: 
            fields = fieldsets
        else:
            tmp = []
            for key in fields:
                if key in fieldsets:
                    tmp.append(key)
            fields = tmp
                     
        #print(conditions, fields)

        return Entry.objects.filter(**conditions).values(*fields)


