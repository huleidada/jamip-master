import xml.etree.cElementTree as ET  
import numpy as np
import pathlib
import os

def str2bool(string):
    if string == 'false':
        return False
    else:
        return True

class Xml:
 
    def __init__(self):
        pass
            
    def basesearch(self,xmlfile,keyword):
        result = None
        for event, elem in ET.iterparse(xmlfile,events=('end',)):
            if event =='end':                                             
                if elem.tag == keyword: 
                    result = elem
                    break                                                   
            elem.clear()     
        if result is None:
            raise RuntimeError("Failed search keyword %s" %keyword)
        return elem

    def date(self,xmlfile):
        result = self.basesearch(xmlfile,'created')
        return result.get('DATE')

    def time(self,xmlfile):
        result = self.basesearch(xmlfile,'created')
        return result.get('TIME')

    def soft(self,xmlfile):
        result = self.basesearch(xmlfile,'creator')
        return result.get('NAME')

    def version(self,xmlfile):
        result = self.basesearch(xmlfile,'creator')
        return result.get('VERSION')

    def nprocs(self,xmlfile):
        result = self.basesearch(xmlfile,'nprocs')
        return int(result.text)

    def calculation(self,xmlfile):
        result = self.basesearch(xmlfile,'calculation')
        return result.text

    def prefix(self,xmlfile):
        result = self.basesearch(xmlfile,'prefix')
        return result.text

    def outdir(self,xmlfile):
        result = self.basesearch(xmlfile,'outdir')
        return result.text

    def etot_conv_thr(self,xmlfile):
        result = self.basesearch(xmlfile,'etot_conv_thr')
        return float(result.text)

    def forc_conv_thr(self,xmlfile):
        result = self.basesearch(xmlfile,'forc_conv_thr')
        return float(result.text)

    def ediff(self,xmlfile):
        return self.etot_conv_thr(xmlfile)

    def ediffg(self,xmlfile):
        return self.forc_conv_thr(xmlfile)

    def press_conv_thr(self,xmlfile):
        result = self.basesearch(xmlfile,'press_conv_thr')
        return float(result.text)

    def functional(self,xmlfile):
        result = self.basesearch(xmlfile,'functional')
        return result.text

    def lsda(self,xmlfile):
        result = self.basesearch(xmlfile,'lsda')
        return str2bool(result.text)

    def noncolin(self,xmlfile):
        result = self.basesearch(xmlfile,'noncolin')
        return str2bool(result.text)

    def spinorbit(self,xmlfile):
        result = self.basesearch(xmlfile,'spinorbit')
        return str2bool(result.text)

    def nbnd(self,xmlfile):
        result = self.basesearch(xmlfile,'nbnd')
        return int(result.text)

    def nelec(self,xmlfile):
        result = self.basesearch(xmlfile,'nelec')
        return float(result.text)

    def occupations(self,xmlfile):
        result = self.basesearch(xmlfile,'occupations')
        return result.text

    def ecutwfc(self,xmlfile):
        result = self.basesearch(xmlfile,'ecutwfc')
        return float(result.text)

    def ecutrho(self,xmlfile):
        result = self.basesearch(xmlfile,'ecutrho')
        return float(result.text)

    def diagonalization(self,xmlfile):
        result = self.basesearch(xmlfile,'diagonalization')
        return result.text

    def conv_thr(self,xmlfile):
        result = self.basesearch(xmlfile,'conv_thr')
        return float(result.text)

    def nk(self,xmlfile):
        result = self.basesearch(xmlfile,'nk')
        return int(result.text)

    def lattice(self,xmlfile):
        result = None
        clear = True
        for event, elem in ET.iterparse(xmlfile,events=('start','end')):
            if elem.tag == "cell": 
                if event == 'start':
                    clear = False
                elif event == 'end':
                    result = []
                    ks = elem.findall('./')
                    for k in ks:
                        result.append(k.text.split())
                    result = np.array(result,dtype=float)
                    break                                                   
            if clear:
                elem.clear()     
        if result is None:
            raise RuntimeError("Failed search keyword cell")
        return result

    def kpoints(self,xmlfile):
        result = None
        clear = True
        for event, elem in ET.iterparse(xmlfile,events=('start','end')):
            if elem.tag == "k_points_IBZ": 
                if event == 'start':
                    clear = False
                elif event == 'end':
                    result = []
                    ks = elem.findall('./k_point')
                    for k in ks:
                        result.append(k.text.split())
                    result = np.array(result,dtype=float)
                    break                                                   
            if clear:
                elem.clear()     
        if result is None:
            raise RuntimeError("Failed search keyword k_points_IBZ")
        return result

    def kpoints_weight(self,xmlfile):
        result = None
        clear = True
        for event, elem in ET.iterparse(xmlfile,events=('start','end')):
            if elem.tag == "k_points_IBZ": 
                if event == 'start':
                    clear = False
                elif event == 'end':
                    result = []
                    ks = elem.findall('./k_point')
                    for k in ks:
                        result.append(k.get('weight'))
                    result = np.array(result,dtype=float)
                    break                                                   
            if clear:
                elem.clear()     
        if result is None:
            raise RuntimeError("Failed search keyword k_points_IBZ")
        return result

    def bands(self,xmlfile):
        clear = True
        eigs = []
        occs = []
        for event, elem in ET.iterparse(xmlfile,events=('start', 'end')):
            if elem.tag == "ks_energies": 
                if event =='start': 
                    clear = False
                elif event == 'end':
                    clear = True
                    # eigenvalues %
                    ks = elem.findall('./eigenvalues')[0]
                    eigs.append(np.array(ks.text.split(),dtype=float))
                    # occupations %
                    ks = elem.findall('./occupations')[0]
                    occs.append(np.array(ks.text.split(),dtype=float))
            elif elem.tag == "band_structure":
                if event == 'end':
                    break
            if clear:
                elem.clear()     
        if len(eigs) == 0:
            raise RuntimeError("Failed search keyword ks_energies")
        result = np.stack((eigs,occs),axis=2).astype(float)
        return result

    def nosym(self,xmlfile):
        result = self.basesearch(xmlfile,'nosym')
        return str2bool(result.text)

    def noinv(self,xmlfile):
        result = self.basesearch(xmlfile,'noinv')
        return str2bool(result.text)

    def uspp(self,xmlfile):
        result = self.basesearch(xmlfile,'uspp')
        return str2bool(result.text)

    def paw(self,xmlfile):
        result = self.basesearch(xmlfile,'paw')
        return str2bool(result.text)

    def total_energy(self,xmlfile):
        result = self.basesearch(xmlfile,'etot')
        return float(result.text)

    def fermi_energy(self,xmlfile):
        result = self.basesearch(xmlfile,'fermi_energy')
        return float(result.text)

    def highestOccupiedLevel(self,xmlfile):
        result = self.basesearch(xmlfile,'highestOccupiedLevel')
        return float(result.text)

    def lowestUnoccupiedLevel(self,xmlfile):
        result = self.basesearch(xmlfile,'lowestUnoccupiedLevel')
        return float(result.text)

    def nelec(self,xmlfile):
        result = self.basesearch(xmlfile,'nelec')
        return float(result.text)

    def get_band(self,xmlfile):
        # ispin
        nk = self.nk(xmlfile)
        nbnd = self.nbnd(xmlfile)
        banddat = self.bands(xmlfile)
        assert banddat.shape == (nk,nbnd,2)
        return banddat

    def get_kpoint(self,xmlfile,weight=False):
        nk = self.nk(xmlfile)
        kpoints = self.kpoints(xmlfile)
        assert len(kpoints) == nk
        if weight:
            kpoints_weight = self.kpoints_weight(xmlfile)
            kpoints = np.stack((kpoints,kpoints_weight),axis=2)
        return kpoints

if __name__ == "__main__":
    xmlfile = "/home/kzhou/QE/example/band/fe.xml"
    a = Xml()
    #tags = ['date','time','soft','version','nprocs','lsda']
    tags = ['kpoints','kpoints_weight']
    #tags = ['total_energy','paw','band2']
    for i in tags:
        print(getattr(a,i)(xmlfile))
        value = getattr(a,i)(xmlfile)
        print(value.shape)
    
