import os
import warnings
warnings.filterwarnings("ignore")

os.environ['DJANGO_SETTINGS_MODULE']='jamip.db.db.settings'

import django
try:
    django.setup()

    from .structure import Structure
    from .composition import Composition
    from .element import Element
    from .species import Species
    from .atom import Atom
    from .entry import Entry
    from .spacegroup import Spacegroup
    from .prototype import Prototype
    from .molStructure import MolStructure
    from .molAtom import MolAtom
    from .molComposition import MolComposition
   
    from ..iostream.read import Read
    from ..iostream.write import Write

except:
    warnings.warn("Django setup failed. Something error in ~/.jamip/bin/jamipdb.")

