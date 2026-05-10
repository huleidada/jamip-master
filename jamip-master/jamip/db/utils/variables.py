# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from enum import Enum

# method 1
#state_of_calculation=Enum('states', ('prepare', 'calculating', 'finished', 'error')) # predefined values of calcualtion's state

# method 2
class state_of_calculation(Enum):
    prepare='prepare'
    calculating='calculating'
    finished='finished'
    error='error'
    unknown='unknown'
    
    
class default_constants(Enum):
    # in spglib
    symprec=1e-5
    angle_tolerance=-1.0
    hall_number=0
    
    # in the calculation of distance
    precision=1e-3
    
    # position of atom
    isNormalizingCoordinate=True
    isContainedConstraints=False
    isContainedVelocities=False
    
# =============================================================================
#     # update data in memory and database
#     isUpdatedInfo=False
#     isPersist=False
# =============================================================================


class environment_variables(Enum):
    # database in setting.py for Django framework
    engine_of_database='django.db.backends.mysql'
    name_of_database='jamipdb'
    user_of_database='root'
    password_of_database='123456'
    host_of_database='localhost'
    port_of_database='3366'
    
    # pseudopotential library for VASP
    pseudopotential_of_VASP='/home/fu/software/pot5.4'
    
# ---------- test ----------
