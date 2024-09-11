import matplotlib.pyplot as plt
import numpy as np
from cosmic.evolve import Evolve
from cosmic.sample.initialbinarytable import InitialBinaryTable

Z = 1.26e-02
BSEDict = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': [1.0, 1.0], 'pts1': 0.05, 
           'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 
           'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 
           'ceflag': 1, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265, 'gamma': -2.0, 'pisn': -2, 
           'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 
           'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 
           'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
           'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 1, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 
           'ussn' : 1, 'sigmadiv' :-20.0, 'qcflag' : 5, 'eddlimflag' : 0, 
           'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,
                             2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
           'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 0, 'ST_tide' : 0, 
           'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.02, 'bhms_coll_flag' : 0, 
           'don_lim' : -1, 'acc_lim' : [-1, -1], 'rtmsflag':0, 'wd_mass_lim' : 1}

paths_h = ['../data/bh1/non-rotating/hydrogen/m15_115/', 
           '../data/bh1/non-rotating/hydrogen/m15_335/',
           '../data/bh1/non-rotating/hydrogen/m15_555/']

paths_he = ['../data/bh1/non-rotating/helium/m15_115/', 
           '../data/bh1/non-rotating/helium/m15_335/',
           '../data/bh1/non-rotating/helium/m15_555/']

labels = ['less overshoot', 'standard', 'more overshoot']

bhms_list = []
bpp_list = []
initC_list = []
bcm_list = []
for p_h, p_he in zip(paths_h, paths_he):
    SSEDict = {'stellar_engine' : 'metisse', 
           'path_to_tracks': p_h,
           'path_to_he_tracks': p_he}

    ngrid = 300
    mass_1 = np.logspace(1, 2, ngrid)
    mass_2 = 0.76 * np.ones(ngrid)
    porb = 550000 *np.ones(ngrid)
    ecc = 0.0 * np.ones(ngrid)
    tphysf = 13700.0*np.ones(ngrid)
    metallicity = Z * np.ones(ngrid)    

    binary_grid = InitialBinaryTable.InitialBinaries(
        m1=mass_1, m2=mass_2, porb=porb, ecc=ecc, 
        tphysf=tphysf, kstar1=1*np.ones(ngrid), kstar2=1*np.ones(ngrid), metallicity=metallicity)  

    bpp, bcm, initC, kick_info  = Evolve.evolve(initialbinarytable=binary_grid, BSEDict=BSEDict, SSEDict=SSEDict, dtp=0.0)

    bhms = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 1) & (bpp.porb > 0)].groupby('bin_num', as_index=False).first()

    bhms_list.append(bhms)
    bpp_list.append(bpp)
    initC_list.append(initC)
    bcm_list.append(bcm)

linestyles = ['-', ':', '--']
for bcm, bpp, initC, label, ls in zip(bcm_list, bpp_list, initC_list, labels, linestyles):
    bpp_select = bpp.loc[bpp.evol_type.isin([3,9])]
    for bn in bpp_select.bin_num.unique():
        print(label)
        print(bpp.loc[bpp.bin_num == bn][['tphys', 'mass_1', 'mass_2', 'kstar_1', 'rad_1', 'evol_type']])
    plt.plot(initC.mass_1, bpp.loc[bpp.kstar_1 > 6].groupby('bin_num', as_index=False).first().mass_1, ls=ls, label=label)
plt.xlabel('initial mass [Msun]')
plt.legend(prop={'size':14})
plt.ylabel('CO mass [Msun]')
plt.tight_layout()
plt.savefig('m15_grid300.png', facecolor='white', dpi=300)

