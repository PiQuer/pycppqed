""" This module provides functions to average a C++QED MCWF trajectory and to save
the results to a file. It is also possible to convert the results to a format readable by matlab. The main
function is :func:`calculateMeans`.
"""

import os
from pycppqed.io import load_cppqed
import numpy as np
import scipy.io
import helpers
import logging

def _genuine_timesteps(evs):
    """ This function determines of how many time steps a trajectory consists. Because of a time-fuzziness in
    the C++QED stepper, small numerical errors in the time steps are possible. For example, if T=10 is chosen,
    a trajectory could have output at T=8,9,10 or at T=7.999,8.999,9.999,10. This functions compares the
    last time step to the next-to-last timestep, and if they are closer then a threshold of 10^-4, the last
    timestep is discarded.    
    """
    if np.allclose(evs[0,-2:-1],evs[0,-1:],atol=1e-4):
        return np.size(evs,1)-1
    else:
        return np.size(evs,1)
    
def _shift_indices(l, s):
    return [i+s for i in l]

def calculateRho(basename,dirname='.'):
    """ This function averages the density matrices in a C++QED MCWF trajectory ensemble. Note that the
    equivalent matlab code is found to be orders of magnitude faster because of multi-threading.
    
    :param basename: String with the basename of the files to import.
    :type basename: str
    :param dirname: Directory name from which files are imported.
    :type dirname: str
    :returns: A tuple `(timevec,rho)`, first entry is the a vector of times at which statevector were printed,
        the second entry is a :class:`np.ndarray` where each row `i` corresponds to a averaged density matrix at
        time `timevec[i]`.
    :retval: tuple 
    """
    filelist = helpers.generate_filelist(basename,dirname)
    (_,qs) = load_cppqed(filelist[0])
    timevec = qs.time
    tdim = qs.shape[0]
    totaldim = np.prod(qs.shape[1:])
    rho = np.zeros((tdim,totaldim,totaldim))
    for f in filelist:
        (_,qs) = load_cppqed(f)
        qs = qs.reshape(tdim,-1)
        rho += qs[:,:,np.newaxis]*np.conj(qs[:,np.newaxis,:])
    return (timevec,rho/len(filelist))
    

def calculateMeans(basename,expvals=[],variances=[],varmeans=[],stdevs=[],stdevmeans=[], datadir='.', outputdir='.', usesaved=True, matlab=True, bz2only=False):
    '''
    Calculate the mean expectation values, mean variances and mean standard deviations from an
    ensemble of C++QED MCWF trajectories. The results are saved to a file.
    
    :param basename:  String with the basename of the files to import, e.g. "RingCavity"
    :type basename: str
    :param expvals: List of indices which contain normal expectation values (Note: first column has index 1)
    :type expvals: list
    :param variances: List of indices which contain variances
    :type expvals: list
    :param varmeans: Corresponding mean values to the variances (if given, this list must have 
            the same length as variances). This can be left empty, in which case the mean values
            are expected in the column just before each variance, respectively.
    :type varmenas: list
    :param stdevs: List of indices which contain standard deviations
    :type stdevs: list
    :param stdevmeans: See `varmeans`
    :type stdevmeans: list
    :param datadir: Directory name from which files are loaded.
    :type datadir: str
    :param outputdir: Directory name where output is written to. If this is `None`, don't write anny output.
    :type outputdir: str
    :param usesaved: If true, only calculate the results if they have not been calculated before, otherwise try to load
        results from the output directory. Previously calculated results are only used if all trajectory files are
        older than the result file and if the set of `expval`, `stdevs`, `variances` etc is identical.
    :type usesaved:
    :param matlab: Also convert results to matlab format and save a .mat file.
    :returns: An array containing the averaged expectation values, standard deviations and variances.
    :rtype: :class:`np.ndarray`
    '''
    
    for l in (expvals,variances,varmeans,stdevs,stdevmeans):
        l[:] = _shift_indices(l, -1)
    
    if variances and varmeans == []:
        varmeans = _shift_indices(variances,-1)
    if stdevs and stdevmeans == []:
        stdevmeans = _shift_indices(stdevs,-1)
    
    # First check if we need to do anything
    filelist = helpers.generate_filelist(basename,datadir,bz2only)
    logging.info("Found %i files."%len(filelist))
    if outputdir:
        datafile = os.path.join(outputdir,basename+".mean.npz")
        matlabfile = os.path.join(outputdir,basename+".mean.mat")
        if os.path.exists(datafile) and usesaved:
            if max(map(os.path.getmtime,filelist))<os.path.getmtime(datafile):
                saved = np.load(datafile)
                if np.array_equal(np.array(expvals),saved['expvals']) \
                    and np.array_equal(np.array(variances),saved['variances']) \
                    and np.array_equal(np.array(varmeans),saved['varmeans']) \
                    and np.array_equal(np.array(stdevs),saved['stdevs']) \
                    and np.array_equal(np.array(stdevmeans),saved['stdevmeans']):
                        logging.info("Using saved file.")
                        return saved['result']

    # initialize result with the zero and the correct shape, the times in the first row
    (evs,_)= load_cppqed(filelist[0])
    result = np.zeros(evs.shape)
    result[0,:]=evs[0,:]
    means = expvals+varmeans+stdevmeans
    for f in filelist:
        logging.debug(f)
        (evs,qs) = load_cppqed(f)
        result[means] += evs[means]/len(filelist)
        result[variances] += (evs[variances]+evs[varmeans]**2)/len(filelist)
        result[stdevs] += (evs[stdevs]**2+evs[stdevmeans]**2)/len(filelist)
        del(evs,qs)
    result[variances] = result[variances]-result[varmeans]**2
    result[stdevs] = np.sqrt(result[stdevs]-result[stdevmeans]**2)
    result = np.transpose(result)
    if outputdir:
        helpers.mkdir_p(outputdir)
        np.savez(datafile,result=result,expvals=np.array(expvals),variances=np.array(variances),
                 varmeans=np.array(varmeans),stdevs=np.array(stdevs),stdevmeans=np.array(stdevmeans),
                 numtraj=np.array(len(filelist)))
        if matlab:
            scipy.io.savemat(matlabfile,{"result":result,"means":np.array(means)+1,"expvals":np.array(expvals)+1,"variances":np.array(variances)+1,
                                         "varmeans":np.array(varmeans)+1,"stdevs":np.array(stdevs)+1,"stdevmeans":np.array(stdevmeans)+1,"numtraj":len(filelist)})
    return result

