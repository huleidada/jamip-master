from jamip.structure import read
from .outcar import GrepOutcar
import numpy as np
import spglib
import pathlib
import os

"""https://github.com/QijingZheng/VaspBandUnfolding"""
# install: 
# pip install git+https://github.com/QijingZheng/VaspBandUnfolding

# constant %
lsoc = False  # if calculation with vasp_ncl
lgam = False  # if calculation with vasp_gam
gam_half = 'x'
RYTOEV   = 13.605826
TPI    = 2 * np.pi
AUTOA    = 0.529177249
HSQDTM = RYTOEV * AUTOA * AUTOA
AUTDEBYE = 2.541746
# HSQDTM    =  hbar**2/(2*ELECTRON MASS)

def setWFPrec(rtag):
    '''
    Set wavefunction coefficients precision:
    TAG = 45200: single precision complex, np.complex64, or complex(qs)
    TAG = 45210: double precision complex, np.complex128, or complex(q)
    '''
    if rtag == 45200:
        return np.complex64
    elif rtag == 45210:
        return np.complex128
    elif rtag == 53300:
        raise ValueError("VASP5 WAVECAR format, not implemented yet")
    elif rtag == 53310:
        raise ValueError("VASP5 WAVECAR format with double precision "
                            +"coefficients, not implemented yet")

def save2vesta(phi=None, poscar='POSCAR', prefix='wfc',
               lgam=False, lreal=False, ncol=10):
    '''
    Save the real space pseudo-wavefunction as vesta format.
    '''
    nx, ny, nz = phi.shape
    print(np.max(phi))
    try:
        pos = open(poscar, 'r')
        head = ''
        for line in pos:
            if line.strip():
                head += line
            else:
                break
        head += '\n%5d%5d%5d\n' % (nx, ny, nz)
    except:
        raise IOError('Failed to open %s' % poscar)

    # Faster IO
    nrow = phi.size // ncol
    nrem = phi.size % ncol
    fmt = "%16.8E"

    psi = phi.copy()
    psi = psi.flatten(order='F')
    psi_h = psi[:nrow * ncol].reshape((nrow, ncol))
    psi_r = psi[nrow * ncol:]

    with open(prefix + '_r.vasp', 'w') as out:
        out.write(head)
        out.write(
            '\n'.join([''.join([fmt % xx for xx in row])
                       for row in psi_h.real])
        )
        out.write("\n" + ''.join([fmt % xx for xx in psi_r.real]))
    if not (lgam or lreal):
        with open(prefix + '_i.vasp', 'w') as out:
            out.write(head)
            out.write(
                '\n'.join([''.join([fmt % xx for xx in row])
                           for row in psi_h.imag])
            )
            out.write("\n" + ''.join([fmt % xx for xx in psi_r.imag]))


class Wavecar:

    def __init__(self, path:str, lsorbit:bool, lgamma:bool):
        self.lsorbit = lsorbit
        self.lgamma = lgamma
        self.path = path
        self.file = open(path,'rb')
        self._wavecar_info()
        self._bands = None
        self._kvecs = None

    @classmethod
    def from_file(cls, path:str, lsorbit=None, lgamma=False):
        p = pathlib.Path(path)
        if p.is_dir():
            p = p / 'WAVECAR'
        if not p.exists():
            raise OSError("WAVECAR not exists!")

        if lsorbit is None:
            lsorbit = GrepOutcar().lsorbit(p.parent)
        return cls(p, lsorbit, lgamma)

    def _wavecar_info(self):
        self.file.seek(0) 
        dump = np.fromfile(self.file, dtype=np.float64, count=3)
        self.recl = int(dump[0])
        self.nspin = int(dump[1])
        self.prec = setWFPrec(int(dump[2]))
        self.file.seek(self.recl)
        dump = np.fromfile(self.file, dtype=np.float64, count=12)
        self.nkpts  = int(dump[0])
        self.nbands = int(dump[1])
        self.encut  = dump[2]
        # real space supercell basis
        self.cell = dump[3:].reshape((3,3)) 
        # reciprocal space supercell basis
        self.rec_cell  = 2*np.pi*np.linalg.inv(self.cell).T
        self.volume = abs(np.linalg.det(self.cell))
       
        # Minimum FFT grid size
        Anorm = np.linalg.norm(self.cell, axis=1)
        CUTOF = np.ceil(np.sqrt(self.encut/RYTOEV)/(TPI/(Anorm/AUTOA)))
        self.ngrid = np.array(2 * CUTOF + 1, dtype=int)
        self.text = self.file.tell()

    @property
    def bands(self):
        if self._bands is None:
            self.get_bands_and_kpoints()
        return self._bands
        
    @property
    def kvecs(self):
        if self._kvecs is None:
            self.get_bands_and_kpoints()
        return self._kvecs
        
    @property
    def kpoints(self):
        return self.kvecs

    def get_bands_and_kpoints(self):

        self.file.seek(self.text) 
        self.nplws = np.zeros(self.nkpts, dtype=int)
        self._kvecs = np.zeros((self.nkpts, 3), dtype=np.float64)
        self._bands = np.zeros((self.nspin, self.nkpts, self.nbands, 2), dtype=float)
 
        for ii in range(self.nspin):
            for jj in range(self.nkpts):
                self.seekRec(ii, jj, 0)
                dump = np.fromfile(self.file, dtype=np.float64, count=4+3*self.nbands)
                if ii == 0:
                    self.nplws[jj] = int(dump[0])
                    self._kvecs[jj] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                self._bands[ii,jj,:,0] = dump[:,0]
                self._bands[ii,jj,:,1] = dump[:,2]
        return self._bands, self._kvecs

    def seekRec(self, ispin:int, ikpt:int ,iband:int, shift=0):

        self.checkIndex(ispin, ikpt, iband)

        # wfc data format: nspin * nkpts * (nbands + 1)
        # Because the basic information of each K point occupies one data block
        # And WAVECAR's basic information also occupies a data block
        rec = 2 + ispin * self.nkpts * (self.nbands+1) + \
              ikpt * (self.nbands+1) + iband + shift
        self.file.seek(rec * self.recl)

    def readBandCoeff(self, ispin, ikpt, iband, norm=False):
        '''
        Read the planewave coefficients of specified KS states.
        '''
        self.seekRec(ispin, ikpt, iband+1)
        dump = np.fromfile(self.file, dtype=self.prec, count=self.nplws[ikpt])
        cg = np.asarray(dump, dtype=np.complex128)
        if norm:
            cg /= np.linalg.norm(cg)
        return cg

    def spectral_weight(self,G):
        assert len(G) == self.nkpts
        SW = []
        for ispin in range(self.nspin):
            sw_spin = []
            for ikpt,g in enumerate(G):
                sw_spin.append(self._spectral_weight(ispin,ikpt,g))
            SW.append(sw_spin)

        return np.array(SW, dtype=float)

    def spectral_function(self, SW, nedos=4000, sigma=0.02):
        SF = np.zeros((self.nspin,nedos,self.nkpts))
        bands = self.bands[:,:,:,0]
        emin = bands.min()
        emax = bands.max()
        e0 = np.linspace(emin -5 * sigma , emax + 5*sigma, nedos)
        LorentzSmearing=lambda x,x0,sigma: 1./ np.pi * sigma**2 / ((x - x0)**2 + sigma**2)

        for ispin in range(self.nspin):
            for ii in range(self.nkpts):
                E_Km = bands[ispin,ii,:]
                P_Km = SW[ispin,ii,:]
                SF[ispin,:,ii] = np.sum(LorentzSmearing(e0[:,None], E_Km[None,:],
                              sigma=sigma)* P_Km[None,:], axis=1)
        return e0, SF
        

    def _spectral_weight(self,ispin,ikpt,g):
        nbands = self.nbands

        Gvalid, Gall = self.get_ovlap_G(ikpt)
        Goffset = Gvalid + g[None, :]  
        # Index of the Gvalid in 3D grid
        GallIndex = Gall % self.ngrid[None, :]
        GoffsetIndex = Goffset % self.ngrid[None, :]

        # 3d grid for planewave coefficients
        wfc_k_3D = np.zeros(self.ngrid, dtype=np.complex128)
        # the weights and corresponding energies
        P_Km = np.zeros(nbands, dtype=float)
        E_Km = np.zeros(nbands, dtype=float)

        for nb in range(nbands):
            # initialize the array to zero, which is unnecessary 
            # since the GallIndex is the same for the same K-point
            # wfc_k_3D[:,:,:] = 0.0

            if self.lsorbit:
                # pad the coefficients to 3D grid
                band_coeff = self.readBandCoeff(ispin, ikpt, nb, norm=True)
                nplw = band_coeff.shape[0] // 2
                band_spinor_coeff = [band_coeff[:nplw], band_coeff[nplw:]]

                # energy
                E_Km[Ispinor, nb] = self.bands[ispin,ikpt,nb,0]
                for Ispinor in range(2):
                    wfc_k_3D[GallIndex[:,0], GallIndex[:,1], GallIndex[:,2]] = band_spinor_coeff[Ispinor]

                    # spectral weight
                    P_Km[nb] = np.linalg.norm(
                                wfc_k_3D[GoffsetIndex[:,0], GoffsetIndex[:,1], GoffsetIndex[:,2]]
                            )**2
            else:
                # pad the coefficients to 3D grid
                band_coeff = self.readBandCoeff(ispin, ikpt, nb, norm=True)
                if lgam:
                    nplw = band_coeff.size
                    tmp  = np.zeros((nplw * 2 - 1), dtype=band_coeff.dtype)
                    # for Gamma version, the coefficients corresponding to G \ne 0
                    # is multiplied by a factor of sqrt(2)
                    band_coeff[1:] /= np.sqrt(2.)
                    tmp[:nplw] = band_coeff
                    tmp[nplw:] = band_coeff[1:].conj()
                    band_coeff = tmp

                wfc_k_3D[GallIndex[:,0], GallIndex[:,1], GallIndex[:,2]] = band_coeff
                # energy
                E_Km[nb] = self.bands[ispin,ikpt,nb,0]
                # spectral weight
                P_Km[nb] = np.linalg.norm(
                            wfc_k_3D[GoffsetIndex[:,0], GoffsetIndex[:,1], GoffsetIndex[:,2]]
                        )**2
            #return np.array((E_Km, P_Km), dtype=float).T
        return P_Km

    def get_ovlap_G(self,ikpt,epsilon=1e-5):
        assert 0<= ikpt < self.nkpts, 'Invalid K-point index!'
        # Reciprocal space vectors of the supercell in fractional unit
        Gvecs = self.gvectors(ikpt)
 
        if lgam:
            nplw = Gvecs.shape[0]
            tmp  = np.zeros((nplw * 2 - 1, 3), dtype=int)
            tmp[:nplw,...] = Gvecs
            tmp[nplw:,...] = -Gvecs[1:,...]            # G' = -G
            Gvecs = tmp
 
        # Shape of Gvecs: (nplws, 3)
        # iGvecs = np.arange(Gvecs.shape[0], dtype=int)
 
        # Reciprocal space vectors of the primitive cell
        gvecs = np.dot(Gvecs, np.linalg.inv(self.trans).T)
        # Deviation from the perfect sites
        gd = gvecs - np.round(gvecs)
        # match = np.linalg.norm(gd, axis=1) < epsilon
        match = np.alltrue(np.abs(gd) < epsilon, axis=1)
 
        return Gvecs[match], Gvecs
        
    def gvectors(self, ikpt, check_consistency=True):
        '''
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        '''
        assert 0 <= ikpt < self.nkpts, 'Invalid kpoint index!'
        kvec = self.kvecs[ikpt]
        if kvec[2] - 0.5 > -1e-8:
            kvec[2] -= 1

        ngrid = self.ngrid
        nplws = self.nplws
        fx = [ii if ii < ngrid[0] // 2 + 1 else ii - ngrid[0]
              for ii in range(ngrid[0])]
        fy = [jj if jj < ngrid[1] // 2 + 1 else jj - ngrid[1]
              for jj in range(ngrid[1])]
        fz = [kk if kk < ngrid[2] // 2 + 1 else kk - ngrid[2]
              for kk in range(ngrid[2])]

        if self.lgamma:
            if gam_half == 'x':
                fx = fx[:ngrid[0] // 2 + 1]
            else:
                fz = fz[:ngrid[2] // 2 + 1]

        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing='ij')).reshape(3,-1)  
        kgrid = np.array([gx,gy,gz], dtype=float).T
        if lgam:
            if gam_half == 'z':
                kgrid = kgrid[(gz>0) | ((gz==0)&(gy>0)) | ((gz==0)&(gy==0)&(gx>=0))]
            else:
                kgrid = kgrid[(gx>0) | ((gx==0)&(gy>0)) | ((gx==0)&(gy==0)&(gz>=0))]

        KENERGY = HSQDTM * np.linalg.norm(np.dot(kgrid + kvec[None,:], self.rec_cell), axis=1)**2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        Gvec = kgrid[np.where(KENERGY < self.encut)[0]]
 
        # Check if the calculated number of planewaves and the one recorded in the
        # WAVECAR are equal
        if check_consistency:

            if Gvec.shape[0] != self.nplws[ikpt]:
                if Gvec.shape[0] * 2 == self.nplws[ikpt]:
                    if not self.lsorbit:
                        raise ValueError('try lsorbit=True')

                elif Gvec.shape[0] == 2 * self.nplws[ikpt]:
                    if not lgam:
                        raise ValueError('try lgamma=True')

                else:
                    raise ValueError('''
                    NO. OF PLANEWAVES NOT CONSISTENT:
                        THIS CODE -> %d
                        FROM VASP -> %d
                           NGRIDS -> %d
                    ''' % (Gvec.shape[0],
                           self.nplws[ikpt] // 2 if self.lsorbit else self.nplws[ikpt],
                           np.prod(ngrid))
                    )

        return np.asarray(Gvec, dtype=int)

    def wfc_r(self, ispin=0, ikpt=0, iband=0,
              gvec=None, Cg=None, ngrid=None,
              rescale=None,
              norm=False, kr_phase=False, r0=[0.0, 0.0, 0.0]):
        '''
        Inputs:
            ispin : spin index of the desired KS states, starting from 1
            ikpt  : k-point index of the desired KS states, starting from 1
            iband : band index of the desired KS states, starting from 1
            gvec  : the G-vectors correspond to the plane-wave coefficients
            Cg    : the plane-wave coefficients. If None, read from WAVECAR
            ngrid : the FFT grid size
            norm  : normalized Cg?
         kr_phase : whether or not to multiply the exp(ikr) phase
               r0 : shift of the kr-phase to get full wfc other than primitive cell
        '''
        from scipy.fftpack import fftfreq, fftn, ifftn

        self.checkIndex(ispin, ikpt, iband)

        if ngrid is None:
            ngrid = self.ngrid.copy() * 2
        else:
            ngrid = np.array(ngrid, dtype=int)
            assert ngrid.shape == (3,)
            assert np.alltrue(ngrid >= self.ngrid), \
                "Minium FT grid size: (%d, %d, %d)" % \
                (self.ngrid[0], self.ngrid[1], self.ngrid[2])

        # By default, the WAVECAR only stores the periodic part of the Bloch
        # wavefunction. In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0). Below, the
        # k-point vector and the real-space grid are both in the direct coordinates.
        if kr_phase:
            r = np.mgrid[0:ngrid[0], 0:ngrid[1], 0:ngrid[2]].reshape((3, np.prod(ngrid))).T / ngrid.astype(float)
            r0 = np.array(r0, dtype=float)
            phase = np.exp(1j * np.pi * 2 * np.sum( self.kvecs[ikpt] * (r + r0), axis=1)).reshape(ngrid)
        else:
            phase = 1.0

        # The default normalization of np.fft.fftn has the direct transforms
        # unscaled and the inverse transforms are scaled by 1/n. It is possible
        # to obtain unitary transforms by setting the keyword argument norm to
        # "ortho" (default is None) so that both direct and inverse transforms
        # will be scaled by 1/\sqrt{n}.

        # default normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        normFac = rescale if rescale is not None else np.sqrt(np.prod(ngrid))

        if gvec is None:
            gvec = self.gvectors(ikpt)

        if self.lgamma:
            if gam_half == 'z':
                phi_k = np.zeros(
                    (ngrid[0], ngrid[1], ngrid[2]//2 + 1), dtype=np.complex128)
            else:
                phi_k = np.zeros(
                    (ngrid[0]//2 + 1, ngrid[1], ngrid[2]), dtype=np.complex128)
        else:
            phi_k = np.zeros(ngrid, dtype=np.complex128)

        gvec %= ngrid[None, :]

        if self.lsorbit:
            wfc_spinor = []
            if Cg:
                dump = Cg
            else:
                dump = self.readBandCoeff(ispin, ikpt, iband, norm)
            nplw = dump.shape[0] // 2

            # spinor up
            phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = dump[:nplw]
            wfc_spinor.append(ifftn(phi_k) * normFac * phase)
            # spinor down
            phi_k[:, :, :] = 0.0j
            phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = dump[nplw:]
            wfc_spinor.append(ifftn(phi_k) * normFac * phase)

            del dump
            return wfc_spinor

        else:
            if Cg is not None:
                phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = Cg
            else:
                phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]
                      ] = self.readBandCoeff(ispin, ikpt, iband, norm)

            if self.lgamma:
                # add some components that are excluded and perform c2r FFT
                raise ValueError("deleted")

            # perform complex2complex FFT
            return ifftn(phi_k * normFac) * phase
  
    def get_dipole_mat(self,ki,kj,norm=True,realspace=False):
        '''
        K = [ispin,ikpt,iband]
        calculate Transition Dipole Moment (TDM) between two KS states.
        If "realspace = False", the TDM will be evaluated in momentum space
        Note: |psi_a> and |psi_b> should be bloch function with the same k vector
        '''
        assert len(ki) == len(kj) == 3, 'Must be three indexes!'
        assert ki[1] == kj[1], 'k-point of the two states differ!'
        self.checkIndex(*ki)
        self.checkIndex(*kj)

        Eki = self.bands[ki[0],ki[1],ki[2],0]
        Ekj = self.bands[kj[0],kj[1],kj[2],0]
        dE = Ekj - Eki

        k0 = self.kvecs[ki[1]]
        G0 = self.gvectors(ikpt=ki[1])
        Gk = np.dot(G0 + k0, self.rec_cell)

        CG_mk = self.readBandCoeff(*ki)
        CG_nk = self.readBandCoeff(*kj)
        ovlap = CG_nk.conj() * CG_mk

        if self.lgamma:
            # G > 0 part
            moment_mat = np.sum(ovlap[:,None] * Gk, axis=0)

            # G < 0 part, C(G) = C(-G).conj()
            moment_mat -= np.sum(
                    ovlap[:,None].conj() * Gk,
                    axis=0)

            # remove the sqrt2 factor added by VASP
            moment_mat /= 2.0

        elif self.lsorbit:
            moment_mat = np.sum(
                ovlap[:, None] * np.r_[Gk, Gk],
                axis=0)
        else:
            moment_mat = np.sum(
                ovlap[:,None] * Gk, axis=0
            )

        dipole_mat = -1j / (dE / (2*RYTOEV)) * moment_mat * AUTOA * AUTDEBYE

        return dE, dipole_mat

    def wave_overlap(self):
        '''
        K = [ispin,ikpt,iband]
        '''
        # assert input KPTS valid %
        edges = self.get_band_edge()
        for ispin in range(self.bands.shape[0]):
            # get edge kpt cbm %
            (ikv, ibv), (ikc, ibc) = edges[ispin]

            # get WAV %
            WAVC = self.readBandCoeff(ispin,ikc,ibc,norm=True)
            WAVV = self.readBandCoeff(ispin,ikv,ibv,norm=True)

            # fill data into ngrid %
            # cbm %
            WAVCALL = np.zeros(self.ngrid, dtype=np.complex)
            CIndex = self.gvectors(ikc) % self.ngrid[None, :]
            WAVCALL[CIndex[:,0], CIndex[:,1], CIndex[:,2]] = WAVC
            # vbm %
            WAVVALL = np.zeros(self.ngrid, dtype=np.complex)
            VIndex = self.gvectors(ikv) % self.ngrid[None, :]
            WAVVALL[VIndex[:,0], VIndex[:,1], VIndex[:,2]] = WAVV

        ovlap_3D = np.abs(WAVCALL) * np.abs(WAVVALL)
        self.ovlap = ovlap_3D

        return np.sum(ovlap_3D)**2

    def elf(self, kptw, ngrid=None):

        # the k-point weights
        kptw = np.array(kptw, dtype=float)
        assert kptw.shape == (self.nkpts,), "K-point weights must be provided \
                                              to calculate charge density!"
        # normalization
        kptw /= kptw.sum()

        if ngrid is None:
            ngrid = self.ngrid * 2
        else:
            ngrid = np.array(ngrid, dtype=int)
            assert ngrid.shape == (3,)
            assert np.alltrue(ngrid >= self.ngrid), \
                "Minium FT grid size: (%d, %d, %d)" % \
                (self.ngrid[0], self.ngrid[1], self.ngrid[2])

        fx = [ii if ii < ngrid[0] // 2 + 1 else ii - ngrid[0]
              for ii in range(ngrid[0])]
        fy = [jj if jj < ngrid[1] // 2 + 1 else jj - ngrid[1]
              for jj in range(ngrid[1])]
        fz = [kk if kk < ngrid[2] // 2 + 1 else kk - ngrid[2]
              for kk in range(ngrid[2])]
        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (ngrid[0], ngrid[1], ngrid[2])
        Dx, Dy, Dz = np.meshgrid(fx, fy, fz, indexing='ij')
        # plane-waves: Cartesian coordinate
        Gx, Gy, Gz = np.tensordot(
            self.rec_cell, [Dx, Dy, Dz], axes=(0, 0))
        # the norm squared of the G-vectors
        G2 = Gx**2 + Gy**2 + Gz**2
        # k-points vectors in Cartesian coordinate
        vkpts = np.dot(self.kvecs, self.rec_cell)

        # normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
        normFac = np.sqrt(np.prod(ngrid) / self.volume)

        # electron localization function
        ElectronLocalizationFunction = []
        # Charge density
        rho = np.zeros(ngrid, dtype=complex)
        # Kinetic energy density
        tau = np.zeros(ngrid, dtype=complex)

        for ispin in range(self.nspin):
            # initialization
            rho[...] = 0.0
            tau[...] = 0.0

            for ikpt in range(self.nkpts):

                # plane-wave G-vectors
                igvec = self.gvectors(ikpt)
                # for gamma-only version, complete the missing -G vectors
                if self.lgamma:
                    tmp = np.array([-k for k in igvec[1:]], dtype=int)
                    igvec = np.vstack([igvec, tmp])
                # plane-wave G-vectors in Cartesian coordinate
                rgvec = np.dot(igvec, self.rec_cell)

                k = vkpts[ikpt]                       # k
                gk = rgvec + k[None, :]               # G + k
                gk2 = np.linalg.norm(gk, axis=1)**2   # | G + k |^2

                for iband in range(self.nbands):
                   
                    # omit the empty bands
                    if self.bands[ispin, ikpt, iband, 1] == 0.0:
                        continue

                    rspin = 2.0 if self.nspin == 1 else 1.0
                    weight = rspin * kptw[ikpt] * self.bands[ispin, ikpt, iband, 1]
                    ########################################
                    # faster
                    ########################################
                    # wavefunction in reciprocal space
                    # VASP does NOT do normalization in elf.F
                    phi_q = self.readBandCoeff(ispin=ispin, ikpt=ikpt,
                                               iband=iband,
                                               norm=False)
                    # pad the missing planewave coefficients for -G vectors
                    if self.lgamma:
                        tmp = [x.conj() for x in phi_q[1:]]
                        phi_q = np.concatenate([phi_q, tmp])
                        # Gamma only, divide a factor of sqrt(2.0) except for
                        # G=0
                        phi_q /= np.sqrt(2.0)
                        phi_q[0] *= np.sqrt(2.0)
                    # wavefunction in real space
                    phi_r = self.wfc_r(ispin=ispin, ikpt=ikpt,
                                       iband=iband,
                                       ngrid=ngrid,
                                       gvec=igvec,
                                       Cg=phi_q) * normFac
                    # grad^2 \phi in reciprocal space
                    lap_phi_q = -gk2 * phi_q
                    # grad^2 \phi in real space
                    lap_phi_r = self.wfc_r(ispin=ispin, ikpt=ikpt,
                                           iband=iband,
                                           ngrid=ngrid,
                                           gvec=igvec,
                                           Cg=lap_phi_q) * normFac

                    # \phi* grad^2 \phi in real space --> kinetic energy density
                    tau += -phi_r * lap_phi_r.conj() * weight

                    # charge density in real space
                    rho += phi_r.conj() * phi_r * weight

            # charge density in reciprocal space
            rho_q = np.fft.fftn(rho, norm='ortho')

            # grad^2 rho: laplacian of charge density
            lap_rho_q = -G2 * rho_q
            lap_rho_r = np.fft.ifftn(lap_rho_q, norm='ortho')

            grad_rho_x = np.fft.ifftn(1j * Gx * rho_q, norm='ortho')
            grad_rho_y = np.fft.ifftn(1j * Gy * rho_q, norm='ortho')
            grad_rho_z = np.fft.ifftn(1j * Gz * rho_q, norm='ortho')

            grad_rho_sq = np.abs(grad_rho_x)**2 \
                + np.abs(grad_rho_y)**2 \
                + np.abs(grad_rho_z)**2

            rho = rho.real
            tau = tau.real
            lap_rho_r = lap_rho_r.real

            Cf = 3./5 * (3.0 * np.pi**2)**(2./3)
            Dh = np.where(rho > 0.0,
                          Cf * rho**(5./3),
                          0.0)
            eps = 1E-8 / HSQDTM
            Dh[Dh < eps] = eps
            # D0 = T + TCORR - TBOS
            D0 = tau + 0.5 * lap_rho_r - 0.25 * grad_rho_sq / rho

            ElectronLocalizationFunction.append(1. / (1. + (D0 / Dh)**2))

        return np.array(ElectronLocalizationFunction)

    def read_unfolding(self):

        KPOINTS = self.path.parent/'KPOINTS'
        GPOINTS = self.path.parent/'GPOINTS'
        if not KPOINTS.exists() or not GPOINTS.exists():
            raise IOError("File not exists!")

        with open(KPOINTS,'r') as f:
            for line in f:
                if line.lstrip()[0].lower() == 'r':
                    break
            KPTS = []
            for line in f:
                if len(line.split()) == 4:
                    KPTS.append(line.split())
            KPTS = np.array(KPTS,dtype=np.float64)[:,:3]

        with open(GPOINTS,'r') as f:
            for line in f:
                if line.lstrip()[0].lower() == 'r':
                    break
            GPTS = []
            for line in f:
                if len(line.split()) == 4:
                    GPTS.append(line.split())
            GPTS = np.array(GPTS,dtype=np.float64)[:,:3].astype(int)

        KPTS = KPTS + GPTS

        return KPTS, GPTS

    def checkIndex(self, ispin, ikpt, iband):
        '''
        Check if the index is valid!
        '''
        assert 0 <= ispin <  self.nspin,  'Invalid spin index!'
        assert 0 <= ikpt  <  self.nkpts,  'Invalid kpoint index!'
        assert 0 <= iband <= self.nbands, 'Invalid band index!'

    def get_band_edge(self, fermi:float=None):
        """
        Extract valence band and conduction band indices. 
        Returns:
        list:[[vb,cb] for each spin]
        """
        if fermi is None:
            fermi = GrepOutcar().fermi(self.path.parent)

        edges = []
        for i,spin in enumerate(self.bands):
            for index in np.arange(spin.shape[1]):
                # If the maximum energy exceeds the Fermi level or empty occupancy
                if max(spin[:,index,0]) > fermi or max(spin[:,index,1]) < 0.001:
                    vb,cb = index-1,index
                    break
            kpt_v = np.argmax(spin[:,vb,0])
            kpt_c = np.argmin(spin[:,cb,0])
            edges.append([[kpt_v, vb],[kpt_c, cb]])
        return edges



# wavecar %
if __name__ == "__main__":
    w = Wavecar.from_file('./')
    w.wavecar()
    pdms = w.get_dipole()
    tdms = np.sum(pdms,axis=1)
    w.plot_tdm(tdms)

