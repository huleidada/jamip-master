"""
copy from https://github.com/hungpham2017/mcu
"""
import pathlib
import numpy as np

class Waveder:

    def __init__(self):
        pass

    @classmethod
    def from_file(cls, path, gamma_only=False):
        path = pathlib.Path(path)
        if path.is_dir():
            path = path / "WAVEDER"
        '''
        from scipy.io import FortranFile
    
        data = FortranFile(path, 'r')
        nb_tot, nbands_cder, nkpts, ispin = data.read_record(dtype= np.int32)
        nodesn_i_dielectric_function = data.read_record(dtype= np.float64)
        wplasmon = data.read_record(dtype= np.float64).reshape(3,3)
        cder = data.read_record(dtype= np.complex64).reshape(ispin,nkpts,nbands_cder,nb_tot,3)
 
        return cder, nodesn_i_dielectric_function, wplasmon
        '''

        with open(path, "rb") as fp:

            def readData(dtype):
                """Read records from Fortran binary file and convert to
                np.array of given dtype."""
                data = b""
                while True:
                    prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
                    data += fp.read(abs(prefix))
                    suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
                    if abs(prefix) - abs(suffix):
                        raise RuntimeError(
                            "Read wrong amount of bytes.\n"
                            "Expected: %d, read: %d, suffix: %d." % (prefix, len(data), suffix)
                        )
                    if prefix > 0:
                        break
                return np.frombuffer(data, dtype=dtype)
 
            nbands, nelect, nk, ispin = readData(np.int32)
            nodes_in_dielectric_function = readData(np.float_)  # nodes_in_dielectric_function
            wplasmon = readData(np.float_)  # wplasmon
            if gamma_only:
                cder = readData(np.float_)
            else:
                cder = readData(np.complex64)
 
            cder_data = cder.reshape((3, ispin, nk, nelect, nbands)).T
            print(cder_data.shape)
 
            return cder_data, nodes_in_dielectric_function, wplasmon


class Wavederf:
    """
    Note: This file is only produced when LOPTICS is true AND vasp has been
    recompiled after uncommenting the line that calls
    WRT_CDER_BETWEEN_STATES_FORMATTED in linear_optics.F
    """

    def __init__(self):
        pass

    @classmethod
    def from_file(cls, path):
        path = pathlib.Path(path)
        if path.is_dir():
            path = path / "WAVEDARF"

        with open(path, 'r') as f:

            head = f.readline()
            ispin, nkpts, nbands_cder = np.int32(head.split())

            data = []
            for line in f:
                # x_real, x_imag, y_real, y_imag, z_real, z_imag
                data.append(np.float64(line.split())[-6:])

            assert len(data) == ispin*nkpts*nbands_cder*nbands_cder
            data = np.array(data).reshape(ispin,nkpts,nbands_cder,nbands_cder)
            # float2complex
            datax = np.complex(data[:,0],data[:,1])
            datay = np.complex(data[:,2],data[:,3])
            dataz = np.complex(data[:,4],data[:,5])
            cder = np.c_[datax,datay,dataz]
            cder = np.concatenate([datax,datay,dataz])
 
            '''
            # the last index of cder for cdum_x,cdum_y,cdum_z
            cder = np.empty([ispin,nkpts,nbands_cder,nbands_cder,3])
  
            line = 1
            for spin in range(ispin):
                for kpt in range(nkpts):
                    for band1 in range(nbands_cder):
                        for band2 in range(nbands_cder):
                            x_real, x_imag, y_real, y_imag, z_real, z_imag = np.float64(data[line].split())[-6:]
                            cdum[spin,kpt,band1,band2] = np.asarray([np.complex(x_real,x_imag), np.complex(y_real,y_imag), np.complex(z_real,z_imag)])
                            line += 1
            '''
  
            return cder

    
   
