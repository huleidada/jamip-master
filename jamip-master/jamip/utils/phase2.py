import os
import re
import time
import numpy as np
import pandas as pd
from typing import Optional, Union
import numpy.linalg as nlg
from dataclasses import dataclass

@dataclass
class dpath:
      
    ''' decomposition path  '''

    output: list
    coeff: list
    energy: int

    def __eq__(self, other):
       
        if isinstance(other, self.__class__):
            if set(self.output) == set(other.output):
                return True
        return False
   
def comb2(n, k):
    """ A fast implementation of itertools.combinations """
    a = np.ones((k, n-k+1), dtype=int)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a

class PhaseAnalysis:
    '''
    Generate multivariate phase diagram, search stable phase and decomposition path.
    '''

    def __init__(self, data:pd.DataFrame, elements: Optional[list]):
        '''
        Input data requirements:
            formula: The chemical formula for structures, you don't have to simplify them.
            energy: The formation energy per atom of structures.
        '''
        self.element = elements
        self.formula = data['formula'].values
        self.energy = data['energy'].values
        self.get_component(data['formula'])

    def get_component(self, symbols):
        '''
        Get the proportion of each element in the structures.
        '''
        import scipy.sparse as ss
        import re

        coords = []

        if self.element != None:
            element = self.element

            for symbol in symbols:
                s = re.findall('([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', symbol) 
                coord = np.zeros(len(element))
                for e,num in s:
                    if num == '': num = 1.0
                    coord[element.index(e)] = float(num)
                coords.append(coord)

        else:
            raise RuntimeError("Unimplemented function! Please add elements list at present.")

        self.__component = np.array(coords)

    @property
    def component(self):
        '''Get the proportion of elements not normalized'''
        return self.__component

    @property
    def normalization_component(self):
        '''Get the proportion of elements not normalized'''
        return self.__component / np.sum(self.__component, axis=1)[:,np.newaxis] 


    @classmethod
    def from_csv(cls, filename, elements):

        df = pd.read_csv(filename)

        return cls(df, elements)


    def convex_hull(self, index=None, axis:int=0):
        """
        Search for stable structures in binary phase space.
        
        params:
           index: Select the specified structure to calculate the convex hull.
           axis: The data axes (elements) used to calculate convex hull.

        return:
           Index of stable structure.
        """
        coords = self.normalization_component

        if index != None:
            index = np.array(index)
            coords = coords[index,axis]
            order = index[np.argsort(coords)]
        else:
            assert len(coords[0]) == 2
            coords = coords[:,axis]
            order = np.argsort(coords)

        coords = np.sort(coords)
        energy = self.energy[order]

        # unique %
        if len(coords) > 2:
            diff = coords[1:] - coords[:-1]
            split = np.arange(len(coords))[np.where(diff>1e-8)] + 1
            min_energy = []
            unorder = []
            uncoords = []
            i = 0
            for s in split:
                min_energy.append(np.min(energy[i:s]))
                unorder.append(order[i:s][np.argmin(energy[i:s])])
                uncoords.append(coords[i])
                i = s
            min_energy.append(np.min(energy[i:]))
            unorder.append(order[i:][np.argmin(energy[i:])])
            uncoords.append(coords[i])
            # rename
            energy = np.asarray(min_energy)
            coords = np.asarray(uncoords)
            order = np.asarray(unorder)
       
        xmax = len(energy)
        actv = np.argmin(energy)
        stable = [actv]
        
        # right ! ->
        while actv != xmax-1 :
            actvx = coords[actv]
            actvy = energy[actv]
            tangs = []
            for i in range(actv+1,xmax):
                tangs.append( (energy[i] - actvy) / (coords[i] - actvx))
            actv += np.argmin(tangs) + 1
            stable.append(actv)
       
        # left ! <-
        actv = stable[0]
        while actv != 0:
            actvx = coords[actv]
            actvy = energy[actv]
            tangs = []
            for i in range(0,actv):
                tangs.append( (energy[i] - actvy) / (coords[i] - actvx))
            actv = np.argmax(tangs)
            stable.insert(0,actv)

        return order[stable]

    def decompose(self, aim_coord, stables, intri=True):
        """ Calculate the decomposition energy of the target structure in phase space. """
        #from itertools import combinations
        #indices = np.array(list(combinations(stables, len(self.element))))

        stables = np.array(stables)
        indices = stables[comb2(len(stables), len(self.element))].T
          
        coords = self.normalization_component[indices]
        mask = np.arange(indices.shape[0])[np.where(np.abs(nlg.det(coords)) > 1e-8)]
        valid_coords = coords[mask,:,:]
        #coeff = np.dot(aim_coord, nlg.inv(valid_coords)) 
        coeff = aim_coord @ nlg.inv(valid_coords)
        mask2 = np.arange(coeff.shape[0])[np.where(np.sum(coeff, axis=1)-1<1e-8)]
        if intri:
            mask_ = mask2[np.where( (np.max(coeff[mask2,:], axis=1)-1<1e-8) & (np.min(coeff[mask2,:], axis=1)>-1e-8) )]
            mask2 = mask_

        indices = indices[mask,:][mask2,:]
        energy = self.energy[indices]
        coeffs = coeff[mask2,:]
        Es = np.sum(energy * coeffs, axis=1) 

        # transfer indices with zero coeff to -1
        indices_ = np.where(coeffs>1e-8, indices, np.full_like(indices, -1))
        sorted_indices = np.sort(indices_, axis=1) 
        unique_indices, mask3 = np.unique(sorted_indices, axis=0, return_index=True)

        # save %
        results = []
        for i in mask3:
            dp = dpath(indices[i], coeffs[i], Es[i])
            results.append(dp)

        return results

    def is_stable(self, index, stables):
        """ Judge whether the structure is stable in phase space. """
        coord = self.normalization_component[index]
        energy = self.energy[index]
        dpaths = self.decompose(coord, stables)
        if np.min([i.energy for i in dpaths]) > energy:
            return True

      
    def triangle_zone(self):
        """
        Search for stable structures in ternary phase space.
        
        return:
           Index of stable structure.
        """
        coords = self.normalization_component
        energy = self.energy
        assert len(coords[0]) == 3

        # Search for structures of binary phase/boundary
        binary = [[], [], []]
        ternary = []
        for i,coord in enumerate(coords):
            is_binary = False
            if energy[i] > 0: continue
            for j,value in enumerate(coord):
                if value == 0:
                    binary[j].append(i)
                    is_binary = True
            if not is_binary:
                ternary.append(i)

        # Search for stable structures of binary phase
        stable = set()
        for i,bin in enumerate(binary):
            if len(bin) == 0: continue
            bin_stable = self.convex_hull(index=bin, axis=(i+1)%3 )
            stable.update(bin_stable)
        stable = list(stable)

        # Search for stable structures in the ternary phase, based on 
        # the stable structures previously found on the boundary.
        tri_stable = []
        for i in ternary:
            if self.is_stable(i, stable):
                stable.append(i)
                tri_stable.append(i)

        # check %
        for i in tri_stable[:-1]:
            stable.remove(i)
            if self.is_stable(i, stable):
                stable.append(i)

        return stable
        

    def convex_hull_diagram(self, stable=None):
        """ Plot a binary phase diagram """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        if stable == None:
            stable = self.convex_hull()

        element = self.element
        coords = self.normalization_component[:,0]
        energy = self.energy
       
        unstable = [ i for i in range(len(energy)) if i not in stable ]
        point_edge = np.array([(coords[i],energy[i]) for i in stable])
        point_float = np.array([(coords[i],energy[i]) for i in unstable])

        plt.figure(figsize=(6,4),dpi=144)
        plt.xlim(0,1)
        ax = plt.gca()
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
        ax.set_xticklabels([element[0],0.2,0.4,0.6,0.8,element[1]])
        plt.ylabel('Delta H [eV/atom]')
        plt.xlabel("%s$_{x}$%s$_{1-x}$" %(element[0],element[1]))
        plt.ylim(np.min(point_edge[:,1])*1.1,0)
        plt.grid()
        plt.plot(point_edge[:,0],point_edge[:,1],c='black',linewidth=1)
        plt.scatter(point_float[:,0],point_float[:,1],marker='o',c='',edgecolors='r')
        plt.scatter(point_edge[:,0],point_edge[:,1],marker='o',c='',edgecolors='g')
        plt.savefig('convex_hull.png')

    def mix_product(self, index, triangle):
        coord = self.normalization_component[index]
        dpaths = self.decompose(coord, triangle, intri=False)
        if len(dpaths) and dpaths[0].energy > self.energy[index]:
            farpoint = triangle[np.argmin(dpaths[0].coeff)]
            #print('far:',farpoint, np.min(dpaths[0].coeff))
            return farpoint
        

    def triangle_line(self, stable=None):
        """
        Search for stable boundary in ternary phase space.
        
        return:
           Index of stable structure tuple.
        """
        from itertools import combinations

        if stable == None:
            stable = self.triangle_zone()

        esort = [stable[i] for i in np.argsort(self.energy[stable])]
        esort = np.array(esort, dtype=int)[::-1]
 
        # 初始三角形为能量最低的三个点,初始线为这三点连线
        triangles = [esort[:3]]
        lines = [{i,j} for i,j in combinations(esort[:3], 2)]
        for i in esort[3:]:
            ntri = []
            dpaths = []
            poplines = []
            for j,tri in enumerate(triangles):
                coord = self.normalization_component[i]
                dpath = self.decompose(coord, tri)
                dpaths.extend(dpath)
                # 如果点在三角形外，计算该点是否能够破坏三角形，此时移除一条边和一个三角形，新增两条边
                if len(dpath) == 0:
                    farpoint = self.mix_product(i,tri) 
                    if farpoint != None:
                        popline = []
                        for v in tri:
                            lines.append({i,v})
                            if v == farpoint: continue
                            if {v,farpoint} in lines:
                                ntri.append((v,farpoint,i))
                            popline.append(v)
                        # pop line %
                        while set(popline) in lines:
                            lines.remove(set(popline))
                    else:
                        ntri.append(tri)

            # 如果点在三角形内，计算该点是否能够破坏三角形，此时移除一个三角形，新增三条边
            for dp in dpaths:
                tri = dp.output
                #print(self.formula[i], np.where(esort==i))
                #print([self.formula[j] for j in dp.output], dp.energy)
                if dp.coeff.min() < 1e-8:
                    vertex = tri[dp.coeff.argmin()]
                    popline = []
                    for v in tri:
                        if v == vertex: continue
                        popline.append(v)
                    if set(popline) in lines:
                        for v in tri:
                            if v == vertex: continue
                            if {vertex,v} in lines:
                                ntri.append((v,vertex,i))
                                lines.append({i,v})
                                lines.append({i,vertex})
                    # pop line %
                    while set(popline) in lines:
                       lines.remove(set(popline)) 
                else:
                    for v1,v2 in combinations(tri, len(tri)-1):
                        if {v1,v2} in lines:
                            ntri.append((v1,v2,i))
                            lines.append({v1,i})
                            lines.append({v2,i})
            triangles = ntri

        unique = []
        for l in lines: 
            if l not in unique:
                unique.append(l)
        #uni = []
        #for t in triangles:
        #    t = set(t)
        #    if t not in uni:
        #       uni.append(t)
        #print(uni, len(uni))
        return unique 
            
        

    def triangle_zone_diagram(self, stable=None):
        """ Plot a ternary phase diagram """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import ternary

        if stable == None:
            stable = self.triangle_zone()

        element = self.element
        coords = self.normalization_component
        energy = self.energy
        unstable = [ i for i in range(len(energy)) if i not in stable ]

        figure, tax = ternary.figure(scale=1)
        figure.set_size_inches(5,5)
        tax.boundary(linewidth=1)
        tax.gridlines(multiple=0.1, color="blue")
        tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset = 0.02)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
       
        # Plot a few different styles with a legend
        tax.scatter(coords[unstable],marker='o',color='w',edgecolors='g',s=20, label="Unstable")
        tax.scatter(coords[stable],marker='o',color='w',edgecolors='r',s=20, label="Stable")
        tax.legend()

        # annotate
        for i in stable:
            tax.annotate(str(i),coords[i],alpha=0.8,c='b',xytext=(-6,-12),textcoords='offset points')
       
        # line
        for a,b in self.triangle_line(stable):
            for ia,ib in zip(coords[a],coords[b]):
                if ia < 1e-8 and ib < 1e-8:
                    continue
            tax.line(coords[a], coords[b], linewidth=1, marker='', color='black', linestyle=":")

        # Set Axis labels and Title
        fontsize = 13.5
        tax.right_corner_label('  '+element[0], fontsize=fontsize)
        tax.top_corner_label(' '+element[1], fontsize=fontsize)
        tax.left_corner_label(element[2]+'    ', fontsize=fontsize)
        
        tax.savefig('triangle_zone.png',dpi=144)

    def decompotion_path_output(self, formula, stable=None, maximum:int=5, per:str='atom', energy=None):
        """
        Print the decomposition path of the input structure.

        params:
           formula: The chemical formula for structures, you don't have to simplify them.
           stable: Index of decomposition phases.
           maximum: Maximum number of decomposition paths printed.
           per: Method of balancing the reaction equation, per atom or per formula
        """
        element = self.element
        if isinstance(formula, str):
            s = re.findall('([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', formula) 
            coord = np.zeros(len(element))
            for e,num in s:
                if num == '': num = 1.0
                coord[element.index(e)] = float(num)
            atom_per_formula = np.sum(coord)
            coord /= atom_per_formula

        elif isinstance(formula, (list,np.ndarray)):
            if len(formula) != len(element):
                raise Exception('The number of elements does not match the length of the input list!')
            atom_per_formula = np.sum(formula)
            coord = np.array(formula) / atom_per_formula
            
        else:
            raise ValueError('Unknown input data type.')

        if stable == None:
            stable = np.arange(len(self.energy))
        result = self.decompose(coord, stable)

        order = np.argsort([i.energy for i in result])

        print('Decomposition Path of {0} (per {1})'.format(str(formula),per))

        record = []
        for i in order:
            dpath = result[i]
            if energy==None:
                string = " Expect energy : %.4f    " %dpath.energy
            else:
                string = " Ehull energy : %.4f    " %(energy-dpath.energy)

            for index, coeff in zip(dpath.output, dpath.coeff):
                if coeff < 1e-8: continue
                energy += self.energy[index] * coeff
                if per == 'atom':
                    string += ' {:8s} {:.4f} '.format(self.formula[index], coeff)
                elif per == 'formula':
                    coeff = coeff * atom_per_formula / np.sum(self.component[index])
                    string += ' {:8s} {:.4f} '.format(self.formula[index], coeff)
            if string not in record:
               print(string)
               record.append(string)
               if len(record) >= maximum:
                   break

class Triangle(PhaseAnalysis):
    '''
    Plot the triangle phase diagram

    Input:
        data: pd.DataFrame, index=(formula, energy per formula)
        element: Element query order, the first two values are shown in the diagram
              such as: ['MA', 'Pb', 'Cl']
    '''

    def __init__(self, data:pd.DataFrame, elements: Optional[list]):

        self.element = elements
        self.formula = data['formula'].values
        self.energy = data['energy'].values
        self.get_component(data['formula'])

    def get_vertices(self):
        from itertools import combinations
        from copy import deepcopy

        assert len(self.element) == 3
        coords = np.concatenate((self.component, np.eye(3)))
        energy = np.concatenate((self.energy, np.zeros(3)))
        index = np.arange(1,len(coords))

        vertices = []
        for ib,ic in combinations(index, 2):
            mat = coords[[0,ib,ic]]
            vec = energy[[0,ib,ic]]
            if abs(nlg.det(mat))<1E-8:
                continue
            try:
                 x = list(nlg.solve(mat, vec))
            except nlg.LinAlgError:
                continue

            inside = True
            for vec, E in zip(coords,energy) :
                if np.inner(vec, x) - E > 1e-8 :
                    inside = False
                    break
            if inside:
                # reordering so that anion first, and then A, B, ... for vertices.sort()
                vertices.append(x)
                #print(self.formula[ib], self.formula[ic])

        return np.array(vertices)

    def get_polygon(self, vertices):

        points = vertices[vertices[:,0].argsort()]

        def cross(o,a,b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        
        # Build lower hull 
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
 
        # Build upper hull
        upper = []
        for p in np.flip(points,axis=0):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
 
        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list. 
        total = lower[:-1] + upper[:]
        return np.array(total)

    def plot(self):

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
       
        coords = self.component
        energy = self.energy

        xmin = energy[0] / coords[0,0]
        ymin = energy[0] / coords[0,1]

        d2c = np.array([coords[0,0], coords[0,1], energy[0]])/ coords[0,2]

        d2coords = np.c_[coords[:,:2], energy]
        d2coords[1:] -= coords[1:,2:] * d2c[np.newaxis,:] 
        d2coords[np.where(d2coords[:,0] < 0)] *= -1

        vertices = self.get_vertices()
        polygon = self.get_polygon(vertices)

        plt.figure(figsize=(8,6),dpi=300)
        plt.tick_params(labelsize=16, direction='in')
        ax = plt.gca()
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.xaxis.set_ticks_position('top')
        ax.spines['top'].set_position(('data', 0))
        ax.spines['top'].set_linewidth(3)
        ax.yaxis.set_ticks_position('right')
        ax.spines['right'].set_position(('data', 0))
        ax.spines['right'].set_linewidth(3) 
        plt.xlim(xmin,0)
        plt.ylim(ymin,0)
        plt.plot([xmin,0],[0,ymin],linewidth=2.5, c='k')
      
        # lines %
        for i in range(1, len(coords)):
            if abs(d2coords[i,1]) < 1e-4: continue
            # intercept 
            dx = d2coords[i,2] / d2coords[i,0]
            dy = d2coords[i,2] / d2coords[i,1]
            # cross
            mat = d2coords[[0,i],:2]
            vec = d2coords[[0,i],2]
            if np.abs(nlg.det(mat)) < 1e-8: continue
            x = nlg.solve(mat, vec)
            if dx > 0 or dx < xmin:
                X = [0, x[0]]
                Y = [dy, x[1]]
            elif dy > 0 or dy < ymin:
                X = [dx, x[0]]
                Y = [0, x[1]]
            else:
                X = [dx, 0]
                Y = [0, dy]
            # latex formula %
            label = ''
            s = re.findall('([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', self.formula[i]) 
            for e,num in s:
                label += e
                if num != '1': label += '$_{%s}$' %num
            plt.plot(X,Y, label=label)

        # fill %
        plt.fill(polygon[:,0], polygon[:,1], c='g')
        plt.plot(polygon[1:,0], polygon[1:,1],marker='o',c='k', lw=0,ms=4,mfc='w')
        

        plt.legend(fontsize=16, loc='lower left', ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(self.formula[0]+'.png')

class PhaseTools:

    def __init__(self):
        self.dataset = {}

    @classmethod
    def formula2coord(cls, formula:str, elements:list):

        s = re.findall('([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', formula) 
        coord = np.zeros(len(elements))
        for e,num in s:
            if num == '': num = 1.0
            coord[elements.index(e)] = float(num)
        coord = coord / np.sum(coord)
        return coord

    @classmethod
    def get_decompotion_path_from_oqmd(cls, formula, return_energy=False):

        from .rester import Oqmd
        
        elements = re.findall('[A-Z][a-z]?', formula) 
        coord = cls.formula2coord(formula, elements)
        products = []
        dpath = []

        t0 = time.time()
        df = Oqmd().get_phase_by_elements(elements, stable=True)
        t1 = time.time()
        print('request finished', t1-t0)
        phase = PhaseAnalysis(data=df, elements=elements) 
        diff = np.sum(np.abs(phase.normalization_component-coord),axis=1)
        stable = np.arange(len(df))
        if np.min(diff) < 1e-6:
            dfid = np.argmin(diff)
            products.append(dfid)
            #dpath.append((phase.formula[dfid], 0.0))
            dpath.append((df.loc[dfid,'name'], 0.0))
            stable = stable[np.where(diff>1e-6)]
        results = phase.decompose(coord, stable)
        t2 = time.time()
        print('phase analysis finished', t2-t1)
        best_dpath = results[np.argmin([i.energy for i in results])]
        for index,coeff in zip(best_dpath.output, best_dpath.coeff):
            if coeff > 1e-8: 
                dpath.append((df.loc[index,'name'], float(round(coeff,6))))
                #dpath.append((phase.formula[index], float(round(coeff,6))))
                products.append(index)
        products = df.loc[products]
        if return_energy:
            return products, dpath, best_dpath.energy
        else:
            return products, dpath
 
    @classmethod
    def get_decompotion_structure_from_oqmd(cls, formula, path:str, interval=5, filename:str='{name}.vasp'):
        from jamip.structure import read, write
        from .rester import Oqmd

        if not os.path.exists(path):
            os.makedirs(path)

        products, dpath = cls.get_decompotion_path_from_oqmd(formula)
        for i in products.index:
            kwargs = products.loc[i].to_dict()
            print(kwargs)
            fname = filename.format(**kwargs) 
            if not os.path.exists(os.path.join(path,fname)):
                structure = Oqmd().get_structure_by_id(products.loc[i,'oqmd_id']) 
                write(structure, os.path.join(path,fname))
                time.sleep(interval)
        return products, dpath
 
    def oqmd_phase(self, stdin:str, stdout=None):
        from jamip.structure import read, write
        from .rester import Oqmd

        if stdout == None: stdout = stdin
        dataset = {}
        products = []
        for filename in os.listdir(stdin):

            if filename.startswith('_'): continue
            print(filename)
            cif = os.path.join(stdin,filename)
            try:
                structure = read(cif)
            except:
                print("Invalid structure file: %s" %filename)
                continue
            formula = structure.get_formula()
            product, dpath = self.get_decompotion_path_from_oqmd(formula)
            products.append(product)
            # save 
            dataset[filename] = {'formula': formula, 'dpath': dpath}

        products = pd.concat(products, axis=0, ignore_index=True)
        produtcs = products.drop_duplicates()
        print(products)
        # download structures %
        for i in products.index:
            oqmd_id = products.loc[i,'oqmd_id']
            name = products.loc[i,'name']
            structure = Oqmd().get_structure_by_id(oqmd_id)
            filename = '_%d_%s.vasp' %(oqmd_id, name)
            write(structure, os.path.join(stdout,filename))

        return dataset

    @classmethod
    def write_yaml(cls, dat, path='phase.yaml'):
        from jamip.utils.logger import dump_yaml
        dump_yaml(dat, path)
