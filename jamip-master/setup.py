import setuptools 
import os

SOURCE = ['install.sh']#, 'JAMIP-V1.1.Manual-Chs.pdf']
for path, dirs, files in os.walk('source'):
    for f in files:
        SOURCE.append(os.path.join(os.path.relpath(path,os.getcwd()),f))

DATAFILE = {
   "jamip.analysis.vasp": ["*.csv", "*.dat"],
   "jamip.structure": ["*.csv","*.yaml","*.json"],
   "jamip.db.iostream": ["sym_ops.yaml"],
   "jamip.ml.utils": ["*.csv"],
}

setuptools.setup(data_files = [('jamipenv', SOURCE)],
                 package_data = DATAFILE,
)
