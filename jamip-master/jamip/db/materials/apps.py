from django.apps import AppConfig
import django

version = int(django.__version__[0])

class MaterialsConfig(AppConfig):
    name = 'materials' if version <= 3 else 'jamip.db.materials'
