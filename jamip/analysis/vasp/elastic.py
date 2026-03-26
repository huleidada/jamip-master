import numpy as np

def get_uvec(vec):    
    """Gets a unit vector parallel to input vector"""  
    l = np.linalg.norm(vec)                        
    if l < 1e-8:                          
        return vec           
    return vec / l    


class ElasticTensor:
    '''https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.054115'''

    def __init__(self, array):
        assert array.shape == (6, 6)
        self.voigt = array
        self.data = array

    # @classmethod
    # def from_strains(cls, strains):
    #     return cls(np.linalg.inv(strains))

    @property
    def compliance_tensor(self):
        """The compliance tensor (in A^3/eV).
        S = inv(C)

        Returns:
            _type_: _description_
        """
        return ElasticTensor(np.linalg.inv(self.voigt))

    @property
    def k_voigt(self):
        """K_v bulk modulus (in eV/A^3).
        Bv = [(C11+C22+C33) + 2(C12+C13+C23)]/9

        Returns:
            _type_: _description_
        """      
        return self.voigt[:3,:3].mean()

    @property
    def k_reuss(self):
        """The K_v bulk modulus (in eV/A^3).

        Returns:
            _type_: _description_
        """        
        return 1.0 / self.compliance_tensor.voigt[:3, :3].sum()
        
    @property
    def k_vrh(self):
        """The K_vrh (Voigt-Reuss-Hill) average bulk modulus (in eV/A^3).

        Returns:
            _type_: _description_
        """        
        return 0.5 * (self.k_voigt + self.k_reuss)

    @property
    def g_voigt(self):
        """The G_v shear modulus (in eV/A^3).
        Gv = [(C11+C22+C33) - (C12+C13+C23) + 3(C44+C55+C66)]/15

        Returns:
            _type_: _description_
        """        
        return (
            2.0 * self.voigt[:3, :3].trace() - np.triu(self.voigt[:3, :3]).sum() + 3 * self.voigt[3:, 3:].trace()
        ) / 15.0
        
    @property
    def g_reuss(self):
        """The G_r shear modulus (in eV/A^3).

        Returns:
            _type_: _description_
        """                
        return 15.0 / (
            8.0 * self.compliance_tensor.voigt[:3, :3].trace()
            - 4.0 * np.triu(self.compliance_tensor.voigt[:3, :3]).sum()
            + 3.0 * self.compliance_tensor.voigt[3:, 3:].trace()
        )
    
    @property
    def g_vrh(self):
        """The G_vrh (Voigt-Reuss-Hill) average shear modulus (in eV/A^3).

        Returns:
            _type_: _description_
        """        
        return 0.5 * (self.g_voigt + self.g_reuss)

    @property
    def y_mod(self):
        """
        Calculates Young's modulus (in SI units) using the
        Voigt-Reuss-Hill averages of bulk and shear moduli

        Returns:
            E_H :
        """
        return 9.0e9 * self.k_vrh * self.g_vrh / (3.0 * self.k_vrh + self.g_vrh)

    @property
    def poisson_ratio(self):        
        return (3.0 * self.k_vrh - 2.0 * self.g_vrh) / (6.0 * self.k_vrh + 2.0 * self.g_vrh)
    
    @property
    def homogeneous_poisson(self):
        """
        returns the homogeneous poisson ratio
        """
        return (1.0 - 2.0 / 3.0 * self.g_vrh / self.k_vrh) / (2.0 + 2.0 / 3.0 * self.g_vrh / self.k_vrh)

    @property
    def universal_anisotropy(self):
        """
        returns the universal anisotropy value
        """
        return 5.0 * self.g_voigt / self.g_reuss + self.k_voigt / self.k_reuss - 6.0
    
    @property
    def born_stability_conditions(self):
        """
        TODO: set complete conditions        
        """
        stable = (self.k_vrh > 0.0) & (self.g_vrh > 0.0)
        return stable
    
    ########################################
    # For 2d crystal, the relationship between the elastic constants and moduli can be given based on the Hooke’s law under the in-plane stress condition
    #
    #  [σxx]   [C11 C12  0 ][εxx]
    #  [σyy] = [C12 C22  0 ][εyy]
    #  [σxy]   [ 0   0  C66][2εxy]
    ########################################
    
    @property
    def mechanical_stability_2d(self):
        """      
        """
        C11 = self.voigt[0, 0]
        C22 = self.voigt[1, 1]
        C12 = self.voigt[0, 1]
        C66 = self.voigt[5, 5]
        if C66 > 0.0 and C11 * C22 - C12 ** 2 > 0.0:
            return True
        else:
            return False
        
    def y_mod_2d(self, theta):
        """     
        https://pubs.acs.org/doi/10.1021/acs.jpcc.7b02582
        """
        C11 = self.voigt[0, 0]
        C22 = self.voigt[1, 1]
        C12 = self.voigt[0, 1]
        C66 = self.voigt[5, 5]
        s = np.math.sin(theta)
        c = np.math.cos(theta)
        sigma = C11 * C22 - C12 **2
        epsilon2 = C11*s**4 + C22*c**4 + ((C11*C22-C12**2)/C66 - 2*C12)*s**2*c**2
        return sigma/epsilon2

    def possion_radio_2d(self, theta):
        """     
        https://pubs.acs.org/doi/10.1021/acs.jpcc.7b02582
        """
        C11 = self.voigt[0, 0]
        C22 = self.voigt[1, 1]
        C12 = self.voigt[0, 1]
        C66 = self.voigt[5, 5]
        s = np.math.sin(theta)
        c = np.math.cos(theta)
        epsilon1 = C12*(c**4+s**4) - (C11 + C22 - (C11*C22 - C12**2)/C66)*c**2*s**2
        epsilon2 = C11*s**4 + C22*c**4 + ((C11*C22-C12**2)/C66 - 2*C12)*s**2*c**2
        return epsilon1/epsilon2

    

if __name__ == "__main__":
    pass
