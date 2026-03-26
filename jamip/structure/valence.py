import pkg_resources
from jamip.utils.logger import load_yaml
from jamip.structure.elementInfo import Element 
import numpy as np
import re 

datafile = pkg_resources.resource_filename(__name__, 'bv.yaml')
bv = load_yaml(datafile)

def get_oxistates(bv):
    oxistates = {}
    for element,value in bv.items():
        max_key = max(value, key=value.get)
        oxistates[element] = max_key
    return oxistates 

oxistates = get_oxistates(bv)
pattern = re.compile(r'([A-Z][a-z]?)(\d*)')

def formula_to_list(formula):
    result = pattern.findall(formula)
    species = []
    numbers = []
    for i,j in result:
        species.append(i)
        n = 1 if j=='' else int(j)
        numbers.append(n)
    return species, numbers

def valence_test(species, valences):
    # 价态判定条件:
    # 负价元素需要比正价元素的电负性大或主族大
    species = [Element.from_symbol(i) for i in species]
    anions = []
    cations = []
    for i,j in enumerate(valences):
        if j > 0: cations.append(species[i])
        elif j < 0: anions.append(species[i])

    for s1 in anions:
        for s2 in cations:
            if s1.X < s2.X and s1.col < s2.col:
#                print(s1.symbol, s1.X, s1.row, s2.symbol, s2.X, s2.row)
                return False
    return True
    
def get_valences(species, numbers, shift=0, fixed={}):
    # 生成器，按价态和的绝对值排序返回全部有效的价态组合
    # 该函数支持species包含重复元素，但由于最终输出为字典形式，并不能正确的输出结果
    valences = []
    for s in species:
        if s in fixed:
           valences.append([fixed[s]])
        else:
           valences.append(list(bv[s].keys()))

    numbers = np.array(numbers)
    mesh = np.array(np.meshgrid(*valences)).reshape(len(species),-1).T
    valence_sum = np.sum(mesh*numbers[None,:], axis=1) + shift
    
    for i in np.argsort(np.abs(valence_sum)):
        if valence_test(species, mesh[i]):
            yield valence_sum[i], {str(key):int(value) for key,value in zip(species, mesh[i])}
             
def get_best_valence(species, numbers, total_charge=0, fixed:dict={}):
    # 按价态和的绝对值和权重进行排序，返回第一个有效的价态组合
    first = None
    results = []
    weights = []
    for valence,species in get_valences(species, numbers, -total_charge, fixed): 
        if first != None and first != valence: break
        weight = np.prod([bv[i][j] for i,j in species.items()])
        results.append(species)
        weights.append(weight)
        first = valence

    return results[np.argmax(weights)]

def get_best_valence_from_formula(formula):
    species, numbers = formula_to_list(formula)
    return get_best_valence(species, numbers)

if __name__ == "__main__":                                                                                           
    formula = "BO3Ba3SnAu"
    species, numbers = formula_to_list(formula)
    print(species, numbers)
    print(get_best_valence(species, numbers))

