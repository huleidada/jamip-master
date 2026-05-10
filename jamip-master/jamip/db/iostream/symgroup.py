import os
#import numpy as np
from monty.design_patterns import cached_class
import yaml
                
"""             
Data from https ://yseto.net/
"""             
def get_sym_ops():
    from importlib.resources import is_resource, open_text
    import jamip.db.iostream as jdi

    if is_resource(jdi,'sym_ops.yaml'):
        with open_text(jdi,'sym_ops.yaml') as f:
            symops = yaml.safe_load(f)
    else:
        raise OSError("Missing data file AM15.csv.")

    return symops 

@cached_class
class SpaceGroup:

    def __init__(self):

        self.symops = get_sym_ops()

    def get_sym_ops(self, number:int):

        if number <= 0 or number > 530:
            raise ValueError("Out of range of H-m symbol!")

        return self.symops[number]

#             No.Pg      HM        SF     No.Crystal System  
pointgroup = [[  1,     '1',      'C1',   1 ], 
              [  2,     '-1',     'Ci',   1 ],
              [  3,     '2',      'C2',   2 ],
              [  4,     'm',      'Cs',   2 ],
              [  5,     '2/m',    'C2h',  2 ],
              [  6,     '222',    'D2',   3 ],
              [  7,     'mm2',    'C2v',  3 ],
              [  8,     'mmm',    'D2h',  3 ],
              [  9,     '4',      'C4',   4 ],
              [ 10,     '-4',     'S4',   4 ],
              [ 11,     '4/m',    'C4h',  4 ],
              [ 12,     '422',    'D4',   4 ],
              [ 13,     '4mm',    'C4v',  4 ],
              [ 14,     '-42m',   'D2d',  4 ],
              [ 15,     '4/mmm',  'D4h',  4 ],
              [ 16,     '3',      'C3',   5 ],
              [ 17,     '-3',     'C3i',  5 ],
              [ 18,     '32',     'D3',   5 ],
              [ 19,     '3m',     'C3v',  5 ],
              [ 20,     '-3m',    'D3d',  5 ],
              [ 21,     '6',      'C6',   6 ],
              [ 22,     '-6',     'C3h',  6 ],
              [ 23,     '6/m',    'C6h',  6 ],
              [ 24,     '622',    'D6',   6 ],
              [ 25,     '6mm',    'C6v',  6 ],
              [ 26,     '-6m2',   'D3h',  6 ],
              [ 27,     '6/mmm',  'D6h',  6 ],
              [ 28,     '23',     'T',    7 ],
              [ 29,     'm-3',    'Th',   7 ],
              [ 30,     '432',    'O',    7 ],
              [ 31,     '-43m',   'Td',   7 ],
              [ 32,     'm-3m',   'Oh',   7 ]]
