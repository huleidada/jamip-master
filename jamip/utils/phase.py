import os
import re
import time
import numpy as np
import pandas as pd
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

    def __init__(self, data:pd.DataFrame, elements: list):
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
        coords = []
        if self.element != None:
            element = self.element

            for symbol in symbols:
                s = re.findall(r'([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', symbol) 
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

        if len(self.element) > 4: 
            return self.fast_decompose(aim_coord, stables, intri)

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

    def fast_decompose(self, aim_coord, stables, intri=True):
        """ 
        Calculate the decomposition energy of the target structure in phase space. 

        Parameters:
            aim_coord (np.ndarray): The target structure in phase space. e.g. [0.1, 0.2, 0.3]
            stables (np.ndarray): The stable phase indices. e.g. [0, 1, 2, 4, 5, 7, 9]
            intri (bool, optional): Whether to consider intrinsic stability. Defaults to True.

        Returns:
            list: A list of decomposition paths.
        """
        from scipy.spatial import ConvexHull

        points = self.normalization_component[stables]
        points = np.c_[points[:,:-1], self.energy[stables]]

        # run convexhull and get face indices
        hull = ConvexHull(points)
        indices = hull.simplices  
          
        # filter face with >0 area
        coords = self.normalization_component[indices]
        mask = np.arange(indices.shape[0])[np.where(np.abs(nlg.det(coords)) > 1e-8)]
        valid_coords = coords[mask,:,:]

        # get decomposition coefficients
        coeff = aim_coord @ nlg.inv(valid_coords)
        # filter face with coefficients_sum == 1 (sometime numpy set 0 to 0.5)
        mask2 = np.arange(coeff.shape[0])[np.where(np.sum(coeff, axis=1)-1<1e-8)]
        # filter face with all coefficients in range [0,1] 
        if intri:
            mask2 = mask2[np.where( (np.max(coeff[mask2,:], axis=1)-1<1e-8) & (np.min(coeff[mask2,:], axis=1)>-1e-8) )]

        # update indices
        indices = indices[mask,:][mask2,:]
        energy = self.energy[indices]
        coeffs = coeff[mask2,:]
        Es = np.sum(energy * coeffs, axis=1) 

        # transfer indices with zero coeff to -1 and sort, then get unique indices. e.g. [0.5A + 0.5B + 0.0C] == [0.5A + 0.5B + 0.0D]
        indices_ = np.where(coeffs>1e-8, indices, np.full_like(indices, -1))
        sorted_indices = np.sort(indices_, axis=1) 
        unique_indices, mask3 = np.unique(sorted_indices, axis=0, return_index=True)

        # save %
        results = []
        for i in mask3:
            dp = dpath(indices[i], coeffs[i], Es[i])
            results.append(dp)

        return results

    def get_unique_compositon_indices(self):
        '''
        Get unique composition indices for the normalized components.
        qhull method can't handle duplicate points.

        Returns:
            np.ndarray: A boolean array indicating unique composition indices.
        '''
        from collections import defaultdict

        coords = []
        maps = defaultdict(list)
        for i,c in enumerate(self.normalization_component):
            for j,coord in enumerate(coords):
                if np.sum(np.abs(coord-c)) < 1e-4:
                    maps[j].append(i)
                    break
            else:
                maps[len(coords)].append(i)
                coords.append(c)

        indices = np.zeros(len(self.normalization_component), dtype=bool)
        for value in maps.values():
            if len(value) == 1:
                indices[value[0]] = True
            else:
                i = value[np.argmin(self.energy[value])]
                indices[i] = True
                #print(self.formula[i], self.energy[value], self.energy[i], i)

        return indices

    def qhull(self, species, stables=None, use_cache=True):
        """ 
        Calculate the decomposition energy of the target structure in phase space. 

        Parameters:
            species (list): The list of species to consider.
            stables (np.ndarray, optional): The stable phases to consider.
            use_cache (bool, optional): Whether to use cached results. if True, it will skip the already calculated Ehull.

        Returns:
            tuple: A tuple containing the filters and stable indices.
        """
        from scipy.spatial import ConvexHull

        # 获取输入元素索引
        indices = []
        for i,e in enumerate(self.element):
            if e in species:
                indices.append(i)
        if len(indices) != len(species):
            raise OSError(f"species {species} not match {self.elements}")
        specie_indices = np.array(indices)
        component = self.normalization_component[:,specie_indices]

        energy = self.energy
        arange = np.arange(len(self.energy)) 

        # 获取稳定相列表 (ConvexHull无法处理重叠点，需要预处理)
        if stables is None:
            stables = self.get_unique_compositon_indices() 
        if not hasattr(self, "Ehull") or self.Ehull is None:
            self.Ehull = np.array([None]*len(energy))

        # 获得与输入元素匹配的组分索引
        filters = np.zeros(len(self.energy), dtype=bool)
        for i,coord in enumerate(component): 
            if 1 - np.sum(coord) < 1e-4:
                filters[i] = True
        stables = filters & stables 

        # 仅有一种元素时，取最小值即可
        if len(species) == 1:
            self.Ehull[filters] = energy[filters] - energy[filters].min()
            stable_indices = arange[filters][np.argmin(energy[filters])] 
            #stables = ~filters
            #stables[arange[stables_indices]] = True
            return filters, stable_indices

        # 包含多种元素时，计算凸包
        else:
            points = component[stables]
            points = np.c_[points[:,:-1], self.energy[stables]]
            #print(points.shape)
            #print(points)

            # 数据过少将无法计算相图，异常处理
            if len(points) < len(species):
                print(f"Warning! Less data for species {species}")
                #print(self.formula[stables])
                #print(self.formula[filters])
                stable_indices = arange[stables]
                self.Ehull[stable_indices] = 0
                for i in arange[filters]: #enumerate(component):
                    if use_cache and self.Ehull[i] != None:
                        continue
                    for j in stable_indices:
                        if i == j: continue
                        if np.sum(np.abs(component[i]-component[j])) < 1e-4:
                            self.Ehull[i] = self.energy[i] - self.energy[j]
                return filters, stable_indices

            # calculate convex hull
            hull = ConvexHull(points)
            coords = component[stables][hull.simplices]
            # 排除上凸包部分 (法向量的能量轴>0)
            simplices = []
            for i, eq in enumerate(hull.equations):
                if eq[-2] < 1e-8:
                    simplices.append(hull.simplices[i])

            # 获取非奇异矩阵的顶点坐标
            mask = np.arange(coords.shape[0])[np.where(np.abs(nlg.det(coords)) > 1e-8)]
            valid_coords = coords[mask,:,:]
            valid_simplices = arange[stables][hull.simplices][mask]
          
            # 获取凸包的顶点/稳定相
            stable_indices = np.unique(arange[stables][simplices])
            self.Ehull[stable_indices] = 0
            #print(arange[stables])
            #print(stable_indices)
         
            # 计算非顶点与凸包表面的系数
            for i in arange[filters]: #enumerate(component):
                if i in stable_indices:
                    continue
                if use_cache and self.Ehull[i] != None:
                    continue
                coord = component[i]
                coeff = coord @ nlg.inv(valid_coords)

                # 判断系数和是否接近1
                #mask2 = np.arange(coeff.shape[0])[np.where(np.sum(coeff, axis=1)-1<1e-8)]
                # 判断系数是否在0和1之间, 获得有效顶点索引
                mask2 = np.arange(coeff.shape[0])[np.where((np.max(coeff, axis=1)-1<1e-8) & (np.min(coeff, axis=1)>-1e-8))]
                indices = valid_simplices[mask2,:]
                energy = self.energy[indices]
                coeffs = coeff[mask2,:]
                self.Ehull[i] = self.energy[i] - np.min(np.sum(energy * coeffs, axis=1))

            return filters, stable_indices

    def is_stable(self, index, stables):
        """ Judge whether the structure is stable in phase space. """
        coord = self.normalization_component[index]
        energy = self.energy[index]
        dpaths = self.decompose(coord, stables)
        if np.min([i.energy for i in dpaths]) > energy:
            return True

    def rebuild_phase(self):
        """
        Rebuild the phase diagram.
        """
        from jamip.structure.atom import Composition

        stables = self.get_unique_compositon_indices() 
        self.Ehull = None

        # groupby
        data = []
        for formula in self.formula:
            comp = Composition.from_formula(formula)
            data.append([len(comp.species), comp.chemsys])
        df = pd.DataFrame(data, columns=['nspecie', 'chemsys'])
        for key,grp in df.groupby(['nspecie','chemsys']):
            print(key, len(grp))
            species = key[1].split('-')
            filters, indices = self.qhull(species, stables)
            stables[filters] = False
            stables[indices] = True
            
        print(self.formula[stables])
        return np.arange(len(self.energy))[stables]

    '''
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
            #if energy[i] > 0: continue
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
    '''

    def convex_hull_diagram(self, stable=None):
        """ Plot a binary phase diagram """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        if stable is None:
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
        elif len(dpaths):
            farpoint = triangle[np.argmin(dpaths[0].coeff)]
            return -farpoint
        

    def triangle_line(self, stable=None):
        """
        Search for stable boundary in ternary phase space.
        
        return:
           Index of stable structure tuple.
        """
        from itertools import combinations
        from scipy.spatial import ConvexHull

        if stable is None:
            stable = self.rebuild_phase()
        print(stable)

        # 获取标准化后的成分
        points = self.normalization_component[stable]
        # 将成分和能量合并
        points = np.c_[points[:,:-1], self.energy[stable]]
        # 计算凸包
        hull = ConvexHull(points)
        # 获取顶点的坐标 shape = (nstable, nspecie, nspecie)
        coords = self.normalization_component[stable][hull.simplices]
        #print(hull.simplices)
        #print(hull.neighbors)
        pairs = []
        for tri in hull.simplices: #neighbors:
            lst = [stable[i] for i in tri]
            for i in range(len(lst)):
                next_i = (i + 1) % len(lst)  # 循环索引，处理首尾连接
                pair = sorted([lst[i], lst[next_i]])  # 对每对值进行排序
                pairs.append(pair)
        pairs = np.unique(pairs, axis=0)
        return pairs

    def triangle_face(self, stable, pairs=None, npair=2):
        """
        Search for stable boundary in ternary phase space.
        
        return:
           Index of stable structure tuple.
        """
        from itertools import combinations
        from scipy.spatial import ConvexHull

        if stable is None:
            stable = self.rebuild_phase()
            
        points = self.normalization_component[stable]
        points = np.c_[points[:,:-1], self.energy[stable]]
        hull = ConvexHull(points)

        # filter faces with input lines
        faces = []
        for tri in hull.simplices: #neighbors:
            nn = 0
            lst = [stable[i] for i in tri]
            for i in range(len(lst)):
                next_i = (i + 1) % len(lst)  # 循环索引，处理首尾连接
                pair = sorted([lst[i], lst[next_i]])  # 对每对值进行排序
                if pair in pairs:
                    nn += 1
            if nn >= npair:
                faces.append(lst)
        faces = np.unique(faces, axis=0)
        return faces

    def triangle_zone_diagram(self, stable=None):
        """ Plot a ternary phase diagram """
        import matplotlib
        matplotlib.use('agg')
        try:
            import ternary
        except ImportError:
            raise ImportError("Please install the 'python-ternary' package to use this feature.")
        
        if len(self.element) != 3:
            raise ValueError("Ternary phase diagram requires exactly 3 elements.")

        if stable is None:
            stable = self.rebuild_phase()

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
        # tax.scatter(coords[unstable],marker='o',color='w',edgecolors='g',s=20, label="Unstable")
        tax.scatter(coords[stable],marker='o',color='w',edgecolors='r',s=20, label="Stable")
        tax.legend()

        # annotate
        for i in stable:
            tax.annotate(str(i),coords[i],alpha=0.8,c='b',xytext=(-6,-12),textcoords='offset points')
       
        # line
        for a,b in self.triangle_line(stable):
            for ia,ib in zip(coords[a], coords[b]):
                if ia < 1e-8 and ib < 1e-8:
                    continue
            print(a,b)
            tax.line(coords[a], coords[b], linewidth=1, marker='', color='black', linestyle=":")

        # Set Axis labels and Title
        fontsize = 13.5
        tax.right_corner_label('  '+element[0], fontsize=fontsize)
        tax.top_corner_label(' '+element[1], fontsize=fontsize)
        tax.left_corner_label(element[2]+'    ', fontsize=fontsize)        
        tax.savefig('triangle_zone.png',dpi=144)

    def quaternary_diagram(self, stable=None, color_by_face=True, plot_face=False, fname='quaternary_zone.png'):
        """ Plot a quaternary phase diagram """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        try:
            from quaternary import quaternary
        except ImportError:
            raise ImportError("Please install the 'python-quaternary' package to use this feature.")

        if stable is None:
            stable = self.rebuild_phase()
        #for i in stable:
        #    print(self.normalization_component[i], self.energy[i])

        element = self.element
        coords = self.normalization_component * 100
        # unstable = [ i for i in range(len(energy)) if i not in stable ]
        is_quaternary = np.all(coords[stable] > 1e-7, axis=1)
        quaternary_indices = stable[is_quaternary]
        face_indices = stable[~is_quaternary]

        fig = plt.figure(figsize=(6,6))
        quat = quaternary(fig)
        quat.set_grid(ticklabelpad=[0.4,0.03,0.25]) # set pad for [bottom,right,left]
        quat.set_label1(element[0], pad=0.02)
        quat.set_label2(element[1])
        quat.set_label3(element[2], pad=0.02)
        quat.set_label4(element[3], pad=0.05)        
        # Set the view angle, default = (azimuth=-80, elevation=20, distance=7)
        quat.ax.azim= -95
        quat.ax.dist= 7
        quat.ax.elev= 8

        face_colors = np.array(['lightskyblue', 'wheat', 'palegreen', 'lightcoral'])

        # Plot face points (1-3 components)
        if color_by_face:
            is_in_border = np.sum(coords[stable] < 1e-7, axis=1) > 1
            border_indices = stable[is_in_border]

            for i in range(4):
                is_in_face = coords[stable,i] < 1e-7
                face_indices = stable[(is_in_face) & (~is_in_border)]
                color = face_colors[i]
                label = '-'.join([self.element[j] for j in range(4) if j != i])                
                print(i, label, is_in_face, is_in_border, (is_in_face) & (~is_in_border))         
                print(i, label, coords[face_indices])
                quat.scatter(coords[face_indices,0],coords[face_indices,1],coords[face_indices,2],marker='o',color=color,edgecolors='blue',s=60, depthshade=True)    
                plt.plot([],[],linestyle='--',marker='o',color=color,mec='blue',markersize=8, label=label)  
            quat.scatter(coords[border_indices,0],coords[border_indices,1],coords[border_indices,2],marker='o',color='azure',edgecolors='blue',s=60, depthshade=True)
            quat.scatter(coords[quaternary_indices,0],coords[quaternary_indices,1],coords[quaternary_indices,2],marker='*',color='red',edgecolors='darkred',s=150, label='-'.join(self.element), depthshade=True)
        else:            
            quat.scatter(coords[face_indices,0],coords[face_indices,1],coords[face_indices,2],marker='o',color='skyblue',edgecolors='blue',s=60, label="Surface Phases", depthshade=True)
            quat.scatter(coords[quaternary_indices,0],coords[quaternary_indices,1],coords[quaternary_indices,2],marker='*',color='red',edgecolors='darkred',s=150, label="Quaternary Phases", depthshade=True)

        # plot lines
        bond_lines = []
        for a,b in self.triangle_line(stable):
            pairs = coords[[a,b]]
            faces = np.sum(pairs,axis=0) > 1e-8
            is_on_face = sum(faces)
            if is_on_face == 3:
                color = face_colors[np.where(faces<1e-8)][0]
                quat.plot(pairs[:,0], pairs[:,1], pairs[:,2], linewidth=1, color=color, linestyle="--", alpha=0.8)
            elif is_on_face == 4:
                color = 'purple'
                quat.plot(pairs[:,0], pairs[:,1], pairs[:,2], linewidth=1, color='purple', linestyle="--", alpha=0.8)
                bond_lines.append([a,b])
            # print(pairs[0], pairs[1], color)

        # plot faces
        if plot_face:
            plotted_interior_label = False # Flag for legend
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            faces = self.triangle_face(stable, bond_lines)
            for face in faces:
                points_4d = coords[face]
                x, y, z = quat.get_xyz(points_4d[:,0], points_4d[:,1], points_4d[:,2])#, points_4d[:,3])
                points_3d = np.array([x, y, z]).T

                label_to_add = None
                if not plotted_interior_label:
                    label_to_add = "Interior Planes"
                    plotted_interior_label = True

                try:
                    polygon = Poly3DCollection([points_3d], label=label_to_add)
                    #polygon = Poly3DCollection([points_3d])#, label=label_to_add)
                    polygon.set_facecolor('mediumpurple')
                    polygon.set_edgecolor('grey')
                    polygon.set_linewidth(0.5)
                    polygon.set_alpha(0.6)
                    quat.ax.add_collection3d(polygon)
                except:
                    pass
                break

        plt.legend()#loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.savefig(fname,dpi=144)
        plt.close()

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
            s = re.findall(r'([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', formula) 
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

        print(f'Decomposition Path of {str(formula)} (per {per})')

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

    def __init__(self, data:pd.DataFrame, elements: list):

        self.element = elements
        self.formula = data['formula'].values
        self.energy = data['energy'].values
        self.get_component(data['formula'])

    def get_vertices(self):
        from itertools import combinations

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
            s = re.findall(r'([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', self.formula[i]) 
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

        s = re.findall(r'([A-Z][a-z]{0,2})([0-9]{0,}\.{0,}[0-9]{0,})', formula) 
        coord = np.zeros(len(elements))
        for e,num in s:
            if num == '': num = 1.0
            coord[elements.index(e)] = float(num)
        coord = coord / np.sum(coord)
        return coord

    @classmethod
    def get_decompotion_path_from_oqmd(cls, formula, return_energy=False):

        from .rester import Oqmd
        
        elements = re.findall(r'[A-Z][a-z]?', formula) 
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
            filename = f'_{oqmd_id}_{name}.vasp'
            write(structure, os.path.join(stdout,filename))

        return dataset

    @classmethod
    def write_yaml(cls, dat, path='phase.yaml'):
        from jamip.utils.logger import dump_yaml
        dump_yaml(dat, path)
