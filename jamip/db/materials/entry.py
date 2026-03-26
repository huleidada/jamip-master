# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models
from ..utils.customField import DictField, NumpyArrayField


class Entry(models.Model):
    """
    Entry of calculation.
    
    Relationships:
        entry
            |-structure
                
    Attributes:
        task
            |-structure: calculated structure in case.
            |-name: name of case. 
            |-calculated_parameters: calculated parameters in case.
            # ---------- properties ----------
            |-energy: calculated energy of structure in case.
            |-energy_per_formula: calculated energy per formula for the structure in case.
            |-energy_per_atom: calculated energy per atom for the structure in case.
            |-pressure: pressure of output in calculation.
            |-bandgap: calculated bandgap for the structure.
            |-bandgap_img: bandgap image.
            |-effective_mass_of_bandside: effective mass of electron ({'vbm':vbm, 'cbm':cbm}).
    """
    # relationship
    structure=models.ForeignKey('Structure', null=True, on_delete=models.PROTECT)
    
    name=models.CharField(primary_key=True, max_length=255) # name of case
    # calculation
    calculated_parameters=models.JSONField(blank=True, null=True)
    path=models.CharField(max_length=255, blank=True, null=True) # path of calculation
    # properties
    energy=models.FloatField(blank=True, null=True)
    energy_per_formula=models.FloatField(blank=True, null=True)
    energy_per_atom=models.FloatField(blank=True, null=True)
    
    # property
    bandgap=models.JSONField(blank=True, null=True) # {'direct': 1.8; 'indirect': 1.2}
    bandgap_img=models.ImageField(blank=True, null=True)
    corrected_bandgap=DictField(blank=True, null=True) # {'HSE06': 2.3; 'HSE06+SOC':1.7; ...}
    effective_mass_of_bandside=DictField(blank=True, null=True) # {'hmass':0.7, 'emass':0.3}
    
    # optical
    optical_bandgap=DictField(blank=True, null=True) # {'kpoint': [xx, xx, xx]; 'nband': xx; 'bandgap': xx}
    dielectric_constant=NumpyArrayField(blank=True, null=True) # 3x3 array
    born_effective_charge=DictField(blank=True, null=True) # {index_of_atom: [3x3]}
    exciton_binding_energy=models.FloatField(blank=True, null=True)
    
    # mechanical
    pressure=models.FloatField(blank=True, null=True)
    stress_tensor=NumpyArrayField(blank=True, null=True)
    elastic_constants=NumpyArrayField(blank=True, null=True)
    bulk_modulus=NumpyArrayField(blank=True, null=True)
    
    # phonon
    Raman_frequencies=NumpyArrayField(blank=True, null=True)
    IR_frequencies=NumpyArrayField(blank=True, null=True)
    
    class Meta:
        app_label='materials'
        db_table='entry'
        default_related_name='entry_set'
        
    def __str__(self):
        return self.name
    
# =============================================================================
#     def destroy(self):
#         """
#         delete this entry from database.
#         """
#         raise ValueError('unimplemented code for deleting entry')
# =============================================================================

    def create(self, name, isPersist=False, **kwargs):
        """
        create case's object.
        
        Arguments:
            name: name of case.
            isPersist: if True, save to database. Conversely, only run in memory.
            
            kwargs:
                structure: structure's object.
        
        Returns:
            case's object.
        """
        from materials.structure import Structure
        
        self.name=name
        
        if 'structure' in kwargs:
            structure=kwargs['structure']
            if isinstance(structure, Structure):
                self.structure=structure
                
        if isPersist:
            self.save()
                
        return self
        
        
        
