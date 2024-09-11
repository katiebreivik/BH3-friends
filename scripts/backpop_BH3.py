import numpy as np
import pandas as pd
import emcee

import paths

from ctypes import *
from scipy.stats import gaussian_kde
import multiprocessing
from cosmic import _evolvebin

from scipy.stats import multivariate_normal

import os
os.environ['MKL_NUM_THREADS'] = '1'

np.set_printoptions(suppress=True)


KICK_COLUMNS = ['star', 'disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
                'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
                'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
                'delta_theta_total', 'omega', 'randomseed', 'bin_num']


def set_flags(params_in):
    #set the flags to the defaults
    flags = dict()

    flags["neta"] = 0.5
    flags["bwind"] = 0.0
    flags["hewind"] = 0.5
    flags["beta"] = 0.125
    flags["xi"] = 0.0
    flags["acc2"] = 1.5
    flags["epsnov"] = 0.001
    flags["eddfac"] = 1.0
    flags["alpha1"] = np.array([1.0, 1.0])
    flags["qcrit_array"] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    flags["lambdaf"] = 0.0
    flags["pts1"] = 0.05
    flags["pts2"] = 0.01
    flags["pts3"] = 0.02
    flags["tflag"] = 0
    flags["ifflag"] = 0
    flags["wdflag"] = 1
    flags["rtmsflag"] = 0
    flags["ceflag"] = 1
    flags["cekickflag"] = 2
    flags["cemergeflag"] = 0
    flags["cehestarflag"] = 0
    flags["bhflag"] = 1
    flags["remnantflag"] = 4
    flags["grflag"] = 1
    flags["bhms_coll_flag"] = 1
    flags["mxns"] = 3.0
    flags["pisn"] = -2
    flags["ecsn"] = 2.5
    flags["ecsn_mlow"] = 1.6
    flags["aic"] = 1
    flags["ussn"] = 1
    flags["sigma"] = 265.0
    flags["sigmadiv"] = -20.0
    flags["bhsigmafrac"] = 1.0
    flags["polar_kick_angle"] = 90.0
    flags["natal_kick_array"] = np.array([[-100.0,-100.0,-100.0,-100.0,0],[-100.0,-100.0,-100.0,-100.0,0.0]])
    flags["kickflag"] = 0
    flags["rembar_massloss"] = 0.5
    flags["bhspinmag"] = 0
    flags["don_lim"] = -2
    flags["acc_lim"] = np.array([-1, -1])
    flags["gamma"] = -2
    flags["bdecayfac"] = 1
    flags["bconst"] = 3000
    flags["ck"] = 1000
    flags["windflag"] = 3
    flags["qcflag"] = 5
    flags["eddlimflag"] = 0
    flags["fprimc_array"] = np.ones(16) * 2.0/21.0 * 0.0
    flags["randomseed"] = 42
    flags["bhspinflag"] = 0
    flags["rejuv_fac"] = 1.0
    flags["rejuvflag"] = 1
    flags["htpmb"] = 3
    flags["ST_cr"] = 0
    flags["ST_tide"] = 0
    flags["zsun"] = 0.02
    flags["using_cmc"] = 0
    natal_kick = np.zeros((2,5))
    qcrit_array = np.zeros(16)
    alpha1 = np.zeros(2)
    qc_list = ["qMSlo", "qMS", "qHG", "qGB", "qCHeB", "qAGB", "qTPAGB", "qHeMS", "qHeGB", "qHeAGB"]
    
    for param in params_in.keys():
        # handle kicks
        # this is hacky -- think about this more.
        if param in ["vk1", "phi1", "theta1", "omega1", "vk2", "phi2", "theta2", "omega2"]:
            if "1" in param:
                if "vk" in param:
                    natal_kick[0,0] = params_in[param]
                elif "phi" in param:
                    natal_kick[0,1] = params_in[param]
                elif "theta" in param:
                    natal_kick[0,2] = params_in[param]
                elif "omega" in param:
                    natal_kick[0,3] = params_in[param]
                natal_kick[0,4] = 1
            elif "2" in param:
                if "vk" in param:
                    natal_kick[1,0] = params_in[param]
                elif "phi" in param:
                    natal_kick[1,1] = params_in[param]
                elif "theta" in param:
                    natal_kick[1,2] = params_in[param]
                elif "omega" in param:
                    natal_kick[1,3] = params_in[param]
                natal_kick[1,4] = 2
        elif param in qc_list:
            ind_dict = {}
            for k, v in zip(qc_list, range(0,10)):
                ind_dict[v] = k
            qcrit_array[ind_dict[param]] = params_in[param]
        elif param in ["alpha1_1", "alpha1_2"]:
            
            if param == "alpha1_2":
                alpha1[1] = params_in[param]
            elif param == "alpha1_1":
                alpha1[0] = params_in[param]

        else:
            flags[param] = params_in[param]


    if np.any(qcrit_array != 0.0):
        flags["qcrit_array"] = qcrit_array   
    if np.any(natal_kick != 0.0):
        # does this need to be flattened?
        flags["natal_kick"] = natal_kick
    if np.any(alpha1 != 0.0):
        flags["alpha1"] = alpha1
    return flags


def evolv2(params_in, stage, params_out):
    # handle initial binary parameters first
    m1 = params_in["m1"] 
    m2 = params_in["m2"]
    m2, m1 = np.sort([m1,m2],axis=0)
    tb = params_in["tb"] 
    e = params_in["e"]
    metallicity = 1.23e-4
    # set the other flags
    flags = set_flags(params_in)
    
    _evolvebin.windvars.neta = flags["neta"]
    _evolvebin.windvars.bwind = flags["bwind"]
    _evolvebin.windvars.hewind = flags["hewind"]
    _evolvebin.cevars.alpha1 = flags["alpha1"]
    _evolvebin.cevars.lambdaf = flags["lambdaf"]
    _evolvebin.ceflags.ceflag = flags["ceflag"]
    _evolvebin.flags.tflag = flags["tflag"]
    _evolvebin.flags.ifflag = flags["ifflag"]
    _evolvebin.flags.wdflag = flags["wdflag"]
    _evolvebin.flags.rtmsflag = flags["rtmsflag"]
    _evolvebin.snvars.pisn = flags["pisn"]
    _evolvebin.flags.bhflag = flags["bhflag"]
    _evolvebin.flags.remnantflag = flags["remnantflag"]
    _evolvebin.ceflags.cekickflag = flags["cekickflag"]
    _evolvebin.ceflags.cemergeflag = flags["cemergeflag"]
    _evolvebin.ceflags.cehestarflag = flags["cehestarflag"]
    _evolvebin.flags.grflag = flags["grflag"]
    _evolvebin.flags.bhms_coll_flag = flags["bhms_coll_flag"]
    _evolvebin.snvars.mxns = flags["mxns"]
    _evolvebin.points.pts1 = flags["pts1"]
    _evolvebin.points.pts2 = flags["pts2"]
    _evolvebin.points.pts3 = flags["pts3"]
    _evolvebin.snvars.ecsn = flags["ecsn"]
    _evolvebin.snvars.ecsn_mlow = flags["ecsn_mlow"]
    _evolvebin.flags.aic = flags["aic"]
    _evolvebin.ceflags.ussn = flags["ussn"]
    _evolvebin.snvars.sigma = flags["sigma"]
    _evolvebin.snvars.sigmadiv = flags["sigmadiv"]
    _evolvebin.snvars.bhsigmafrac = flags["bhsigmafrac"]
    _evolvebin.snvars.polar_kick_angle = flags["polar_kick_angle"]
    _evolvebin.snvars.natal_kick_array = flags["natal_kick_array"]
    _evolvebin.cevars.qcrit_array = flags["qcrit_array"]
    _evolvebin.mtvars.don_lim = flags["don_lim"]
    _evolvebin.mtvars.acc_lim = flags["acc_lim"]
    _evolvebin.windvars.beta = flags["beta"]
    _evolvebin.windvars.xi = flags["xi"]
    _evolvebin.windvars.acc2 = flags["acc2"]
    _evolvebin.windvars.epsnov = flags["epsnov"]
    _evolvebin.windvars.eddfac = flags["eddfac"]
    _evolvebin.windvars.gamma = flags["gamma"]
    _evolvebin.flags.bdecayfac = flags["bdecayfac"]
    _evolvebin.magvars.bconst = flags["bconst"]
    _evolvebin.magvars.ck = flags["ck"]
    _evolvebin.flags.windflag = flags["windflag"]
    _evolvebin.flags.qcflag = flags["qcflag"]
    _evolvebin.flags.eddlimflag = flags["eddlimflag"]
    _evolvebin.tidalvars.fprimc_array = flags["fprimc_array"]
    _evolvebin.rand1.idum1 = flags["randomseed"]
    _evolvebin.flags.bhspinflag = flags["bhspinflag"]
    _evolvebin.snvars.bhspinmag = flags["bhspinmag"]
    _evolvebin.mixvars.rejuv_fac = flags["rejuv_fac"]
    _evolvebin.flags.rejuvflag = flags["rejuvflag"]
    _evolvebin.flags.htpmb = flags["htpmb"]
    _evolvebin.flags.st_cr = flags["ST_cr"]
    _evolvebin.flags.st_tide = flags["ST_tide"]
    _evolvebin.snvars.rembar_massloss = flags["rembar_massloss"]
    _evolvebin.metvars.zsun = flags["zsun"]
    _evolvebin.snvars.kickflag = flags["kickflag"]
    _evolvebin.se_flags.using_sse = False
    _evolvebin.se_flags.using_metisse = True
    
    
    # setup the inputs for _evolvebin
    zpars = np.zeros(20)
    mass = np.array([m1,m2])
    mass0 = np.array([m1,m2])
    epoch = np.array([0.0,0.0])
    ospin = np.array([0.0,0.0])
    tphysf = 13700.0
    dtp = 0.0
    rad = np.array([0.0,0.0])
    lumin = np.array([0.0,0.0])
    massc = np.array([0.0,0.0])
    radc = np.array([0.0,0.0])
    menv = np.array([0.0,0.0])
    renv = np.array([0.0,0.0])
    B_0 = np.array([0.0,0.0])
    bacc = np.array([0.0,0.0])
    tacc = np.array([0.0,0.0])
    tms = np.array([0.0,0.0])
    bhspin = np.array([0.0,0.0])
    tphys = 0.0
    bkick = np.zeros(20)
    bpp_index_out = 0
    bcm_index_out = 0
    kick_info_out = np.zeros(34)
    kstar = np.array([1,1])
    kick_info = np.zeros((2, 17))

    path_to_tracks = paths.data / 'BH3/hydrogen/w_00/p15_115/'
    path_to_he_tracks = paths.data / 'BH3/helium/w_00/p15_115/'

    #path_to_tracks = ''
    #path_to_he_tracks = ''
    [bpp_index, bcm_index, kick_info_out] = _evolvebin.evolv2(kstar,mass,tb,e,metallicity,tphysf,
                                                              dtp,mass0,rad,lumin,massc,radc,
                                                              menv,renv,ospin,B_0,bacc,tacc,epoch,tms,
                                                              bhspin,tphys,zpars,bkick,kick_info,
                                                              path_to_tracks,path_to_he_tracks)


    bcm = _evolvebin.binary.bcm[:bcm_index].copy()
    bpp = _evolvebin.binary.bpp[:bpp_index].copy()

    bpp = np.hstack((bpp, np.ones((bpp.shape[0], 1))*np.random.uniform(0,100000000)))
    bpp = pd.DataFrame(bpp, 
                       columns=['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 
                                'sep', 'porb', 'ecc', 'RRLO_1', 'RRLO_2', 'evol_type', 
                                'aj_1', 'aj_2', 'tms_1', 'tms_2', 'massc_1', 'massc_2', 
                                'rad_1', 'rad_2', 'mass0_1', 'mass0_2', 'lum_1', 'lum_2', 
                                'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1', 'menv_2', 
                                'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1', 
                                'B_2', 'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'epoch_1', 
                                'epoch_2', 'bhspin_1', 'bhspin_2', 'bin_num'])
    
    kick_info = kick_info_out.reshape(17,2).T
    kick_info = pd.DataFrame(kick_info,
                             columns = ['star', 'disrupted', 'natal_kick', 
                                        'phi', 'theta', 'mean_anomaly',
                                        'delta_vsysx_1', 'delta_vsysy_1', 
                                        'delta_vsysz_1', 'vsys_1_total',
                                        'delta_vsysx_2', 'delta_vsysy_2', 
                                        'delta_vsysz_2', 'vsys_2_total',
                                        'delta_theta_total', 'omega', 
                                        'randomseed'])
    
    
    
    if stage == 'BNS_merger':
        out = bpp.loc[(bpp.kstar_1 == 13) & (bpp.kstar_2 == 13) & (bpp.evol_type == 3)]
    elif stage == 'NSBH_merger':
        out = bpp.loc[((bpp.kstar_1 == 14) & (bpp.kstar_2 == 13) & (bpp.evol_type == 3)) |
                      ((bpp.kstar_1 == 13) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3))]
    elif stage == "BBH_merger":
        out = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)]
    elif stage == "BH_MS":
        out = bpp.loc[((bpp.kstar_1 == 14) & (bpp.kstar_2.isin([0,1])) & (bpp.sep > 0)) |
                      ((bpp.kstar_1.isin([0,1])) & (bpp.kstar_2 == 14) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
        
    elif stage == "NS_MS":
        out = bpp.loc[((bpp.kstar_1 == 13) & (bpp.kstar_2.isin([0,1])) & (bpp.sep > 0)) |
                      ((bpp.kstar_1.isin([0,1])) & (bpp.kstar_2 == 13) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
    elif stage == "WD_MS":
        out = bpp.loc[((bpp.kstar_1.isin([10,11,12])) & (bpp.kstar_2.isin([0,1])) & (bpp.sep > 0)) |
                      ((bpp.kstar_1.isin([0,1])) & (bpp.kstar_2.isin([10,11,12])) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
    elif stage == "BH_GS":
        out = bpp.loc[((bpp.kstar_1 == 14) & (bpp.kstar_2 == 3) & (bpp.sep > 0)) |
                      ((bpp.kstar_1 == 3) & (bpp.kstar_2 == 14) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
    elif stage == "NS_GS":
        out = bpp.loc[((bpp.kstar_1 == 13) & (bpp.kstar_2 == 3) & (bpp.sep > 0)) |
                      ((bpp.kstar_1 == 3) & (bpp.kstar_2 == 13) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
    elif stage == "WD_GS":
        out = bpp.loc[((bpp.kstar_1.isin([10,11,12])) & (bpp.kstar_2 == 3) & (bpp.sep > 0)) |
                      ((bpp.kstar_1 == 3) & (bpp.kstar_2.isin([10,11,12])) & (bpp.sep > 0))].groupby('bin_num', as_index=False).first()
    else:
        raise ValueError("you've not specified one of the available choices for backpop stages. choose from: BNS_merger,NSBH_merger,BBH_merger,BH_MS,NS_MS,WD_MS,BH_GS,NS_GS,WD_GS")

    if len(out) > 0:
        return out[params_out], bpp, kick_info
    else:
        return [0]*len(params_out), 0, 0

#Porb: 185.63 +/- 0.05
#Eccentricity: 0.45  +/- 0.01 
#MBH: 9.80  +/- 0.20
#Mstar: 0.93 +/- 0.05
mean = np.array([32.7, 0.76, 4253.1, 0.7291])
cov = np.array([[0.82**2, 0, 0, 0], [0, 0.05**2, 0, 0], [0, 0, 98.5**2, 0], [0, 0, 0, 0.005**2]])
rv = multivariate_normal(mean, cov)
#m1, m2, tb, e, alpha, vk1, theta1, phi1, omega1
m1lo = 55.0
m2lo = 0.73
tblo = 100.0
elo = 0.0
#alphalo_1 = 0.1
vklo = 10.0
thetalo = 0.0
philo = -90.0
#omegalo = 0.0

m1hi = 70.0
m2hi = 0.79
tbhi = 1000.0
ehi = 0.4
#alphahi_1 = 20.0
vkhi = 55.0
thetahi = 360.0
phihi = 90.0
#omegahi = 360

#m1, m2, logtb, e, alpha_1, vk1, theta1, phi1, omega1
param_names = ["m1", "m2", "tb", "e", "vk1", "theta1", "phi1"]
lower_bound = np.array([m1lo, m2lo, tblo, elo, vklo, thetalo, philo])
upper_bound = np.array([m1hi, m2hi, tbhi, ehi, vkhi, thetahi, phihi])

def likelihood(params_in, stage, params_out):
    for c,i in zip(params_in.keys(), range(len(params_in))):
        if c in ["m1" "m2", "tb", "e", "vk1"]:
            if (params_in[c]<0):
                return -np.inf
            if c=="e":
                if params_in[c] > 1.0:
                    return -np.inf
        if c == "theta1":
            if params_in[c] > thetahi or params_in[c] < thetalo:
                return -np.inf
        if c == "phi1":
            if params_in[c] > phihi or params_in[c] < philo:
                return -np.inf
        if c == "m1":
            if params_in[c] > m1hi or params_in[c] < m1lo:
                return -np.inf
    result = evolv2(params_in, stage, params_out)
    fit_coord = result[0]
    if np.any(fit_coord) == 0.0: return -np.inf
    return rv.logpdf(fit_coord) 

n_dim = len(lower_bound)
n_walkers = 50
p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))


n_steps = 1000

stage="BH_MS"
params_out = ["mass_1", "mass_2", "porb", "ecc"]
params_in = pd.DataFrame(p0, columns=param_names)

#with multiprocessing.get_context("fork").Pool() as pool:
sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood, args=[stage, params_out], parameter_names=param_names)
sampler.run_mcmc(params_in, n_steps, progress=True)



np.savez(paths.data / 'BH3/bh3_p15_115',nwalkers=n_walkers,n_steps=n_steps,chain=sampler.chain)
