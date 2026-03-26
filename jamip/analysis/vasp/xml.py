import xml.etree.cElementTree as ET
import pathlib
import numpy as np

def xml2array(root, dtype=float):
    dim={}
    dim_order=[]
    field = []
    array = []
    shape = []
    for element in root:
        if element.tag == 'dimension':
            dim[element.text] = 0
            dim_order.append(element.text)
        if element.tag == 'field':
            field.append(element.text)
        if element.tag == 'set':
            for text in element.findall(".//"):
                if text.tag == 'set':
                    dim[text.attrib['comment'].split()[0]]+=1
                elif text.tag == 'rc':
                    array.append([c.text for c in text])
                elif len(text.text.split()) == len(field):
                    array.append(text.text.split())
    product=1
    for i in reversed(dim_order[1:]):
         shape.append(int(dim[i]/product))
         product=dim[i]
    shape.extend([-1,len(field)])
    array=np.array(array,dtype=dtype).reshape(shape)
    return array

def xml2varray(root):
    varray=[]
    for i in root:
        varray.append(i.text.split())
    return np.array(varray)


class Xml:

    def __init__(self, path):

        self.xmlfile = path
        if not self.xmlfile.exists():
            raise OSError("XMLFile not exists!")

    @property
    def xmlfile(self):
        return self._xmlfile

    @xmlfile.setter
    def xmlfile(self, path:str):
        path = pathlib.Path(path)
        if path.is_dir():
            path = path/'vasprun.xml'
        self._xmlfile = path

    def _get_band(self):
 
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        bands=root.findall("./calculation/eigenvalues/array")[0]
        bands_array=xml2array(bands, dtype=float)
        return bands_array

    def _get_kpoint(self):

        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        kpointlist=root.findall("./kpoints/varray[@name='kpointlist']")[0]
        kpoints=xml2varray(kpointlist).reshape(-1,3).astype(float)
        return kpoints

    def _get_total_dos(self):

        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        dos=root.findall("./calculation/dos/total/array")[0]
        dos_array=xml2array(dos, dtype=float)
        return dos_array

    def _get_partial_dos(self):

        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        dos=root.findall("./calculation/dos/partial/array")[0]
        dos_array=xml2array(dos).astype(float)
        return dos_array

    def fermi_energy(self):
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        efermi = root.findall("./calculation/dos/i[@name='efermi']")[0].text.strip()
        return float(efermi)

    def elements(self):
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        elements = root.findall("./atominfo/array[@name='atoms']")[0]#.text.strip()
        elements = xml2array(elements, dtype=str)[:,0]
        return elements

    def mass(self):
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        data = root.findall("./atominfo/array[@name='atomtypes']")[0]#.text.strip()
        masses = []
        lines = xml2array(data, dtype=str)#[:,0]
        for row in lines:
            masses += [float(row[2])] * int(row[0])
        return np.array(masses, dtype=float)

    def volume(self):
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        efermi = root.findall("./structure[@name='finalpos']/crystal/i[@name='volume']")[0].text.strip()
        return float(efermi)

    def reciprocal_lattice_vectors(self):
        tree = ET.ElementTree(file=self.xmlfile)
        root = tree.getroot()
        rec_basis=root.findall("./structure[@name='initialpos']/crystal/varray[@name='rec_basis']")[0]
        rec_vector=xml2varray(rec_basis).astype(float)
        return rec_vector

    def _get_forces(self):

        forces = None
        clear = True
        for event, elem in ET.iterparse(self.xmlfile,events=('start','end')):
            if event == 'start':
                if elem.tag == "varray":
                    clear = False

            elif event == 'end':
                if elem.get('name') == "forces":
                    forces = xml2varray(elem)
                    elem.clear()
                    break

            if clear:
                elem.clear()

        if forces is None:
            raise OSError("Read FORCES failed.")

        return forces

    def _get_hessian(self):

        hessian = None
        clear = True
        for event, elem in ET.iterparse(self.xmlfile,events=('start','end')):
            if event == 'start':
                if elem.tag == "varray":
                    clear = False

            elif event == 'end':
                if elem.get('name') == "hessian":
                    hessian = xml2varray(elem)
                    elem.clear()
                    break

            if clear:
                elem.clear()

        if hessian is None:
            raise OSError("Read hessian failed.")

        # reshape (natom*3, natom*3) -> (natom, natom, 3, 3)
        natom = int(hessian.shape[0] / 3)
        hessian = hessian.reshape(natom,3,natom,3).transpose(0,2,1,3).astype('float')

        return hessian

    def get_dielectric_func(self):

        clear = True
        imag = real = None
        for event, elem in ET.iterparse(self.xmlfile,events=('start','end')):
            if event == 'start':
                if elem.tag == "dielectricfunction":
                    clear = False

            elif event == 'end':
                if elem.tag == "imag":
                    array = elem.findall('./array')[0]
                    imag = xml2array(array)
                    elem.clear()
                elif elem.tag == "real":
                    array = elem.findall('./array')[0]
                    real = xml2array(array)
                    elem.clear()
                elif elem.tag == "dielectricfunction":
                    break

            if clear:
                elem.clear()

        if imag is None or real is None:
            raise RuntimeError("Failed search keyword 'dielectricfunction'")
        return imag,real
