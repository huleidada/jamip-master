from jamip.utils.plot import globalvar as gl

gl.band.emin = -1
gl.band.emax = 3
gl.band.xlabel = ''
gl.band.ylabel = 'Energy (eV)'

gl.dos.emin = -1
gl.dos.emax = 3
gl.dos.xlabel = ''
gl.dos.ylabel = 'Energy (eV)'

if __name__ == '__main__':
    from jamip.utils.plot import Plot
    pl = Plot(path='Results/Si.vasp')
    pl.plots('fatband')
    #pl.plots('band')
    #pl.plots('pdos')
    #pl.plots('unfolding')

