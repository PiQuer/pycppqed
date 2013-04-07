""" This module defines some helper functions.
"""

import os
import errno
import pycppqed as qed
import warnings
import numpy as np
import itertools
import base64
import cPickle as pickle
import logging
import sys

def ignore_warnings():
    warnings.simplefilter("ignore",FutureWarning)
    warnings.simplefilter("ignore",DeprecationWarning)

def import_class(s):
    r"""Import a class specified by the string s.

    :param s: The name of the class to import, e.g. 'mypackage.mymodule.myclass'
    :returns: The class.
    """
    components = s.split('.')
    modulename = '.'.join(components[:-1])
    classname = components[-1]
    module = __import__(modulename, globals(), locals(), [classname])
    return getattr(module,classname)

def mkdir_p(path):
    """http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError, exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise
        
def rm_f(path):
    """Remove a file if it exists, otherwise do nothing
    """
    try:
        os.remove(path)
    except OSError, exc:
        if exc.errno == errno.ENOENT:
            pass
        else: raise

def replace_dirpart(path,newdir):
    return os.path.join(newdir,os.path.basename(path))
     
def check_if_file_exists(basename,extension):
    if os.path.exists(basename):
        return (basename,False)
    elif os.path.exists(basename+extension):
        return (basename+extension,True)
    else:
        return (None,False)

def generate_filelist(basename,dirname,bz2only=False):
    """ This function generates a list of files which start with `basename` and either end with a digit (possibly with an `.bz2` extension)
    
    :param basename: The files have to begin with this string.
    :type basename: str
    :param dirname: Search in this directory.
    :type dirname: str
    :returns: A list of matching filenames.
    :param bz2only: Only consider files ending in bz2
    :type bz2only: bool
    :retval: list
    """
    filelist = [ os.path.join(dirname,x) for x in os.listdir(dirname) if x.startswith(basename) and 
                        ((not x.endswith('.bz2') and x[-1].isdigit()) or (x.endswith('.bz2') and x[-5].isdigit() ))]
    assert filelist != []
    if bz2only: filelist = [f for f in filelist if f.endswith('.bz2')]
    return filelist


def product(*args, **kwds):
    """ This is an implementation of :class:`itertools.product`, which is missing in Python 2.4.
    """
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
        
        
def matlab_range_to_list(s):
    """ Generate a list from a matlab-style range definition.
    
    :param s: A range definition of the form `start:[step:]stop`. Only integers are supported, and `stop` is included.
    :type s: str
    :returns: A list `[start,...,stop]`
    :retval: list
    """
    spec = map(float,s.split(':'))
    if len(spec)==3:
        start,step,stop=spec
    elif len(spec)==2:
        start,stop=spec
        step=1
    else:
        raise ValueError('%s: range specification has to be start:[step:]stop'%s)
    stop+=step
    return map(_int_if_int,list(np.arange(start,stop,step)))

def matlab_range_to_string(s,sep=";"):
    return sep.join(map(str,matlab_range_to_list(s)))
    
def cppqed_t(filename):
    r"""This helper function returns the last timestep t of a C++QED file.
    
    :param filename: The name of the file to load.
    :type filename: str
    :returns: The last timestep `T` of the trajectory or `None` if the file could
        not be loaded.
    :retval: :class:`numpy.float64`
    """
    try:
        evs, svs = qed.load_cppqed(filename)
    except:
        return None
    return evs[0,-1]

  
def _int_if_int(i):
    return int(i) if int(i)==i else i

def range_str(start,stop,step,sep=";"):
    return sep.join(map(str,map(_int_if_int,list(np.arange(start,stop,step)))))

def retrieveObject(argv):
    if not len(argv)>1:
        logging.error("Need a JobArray object as commandline argument. "+\
                      "Note that this script is not intended to be called manually.")
        sys.exit(1)
    job = pickle.loads(base64.decodestring(argv[1]))
    logging.getLogger().setLevel(job.loglevel)
    return job

class VariableParameters(object):
    def __init__(self, parameterValues=dict(), parameterGroups=(), combine=True):
        self.parameterValues = parameterValues
        if not combine:
            self.parameterGroups = (parameterValues.keys(),)
        else:
            self.parameterGroups = parameterGroups
        self._checkParameterGroups()
    def subdir(self, parSet, numeric=False):
        if numeric:
            return "%02d"%(sorted(list(self.parGen())).index(parSet)+1)
        else:
            return '_'.join(["%s=%s"%i for i in parSet.items()])
    def _checkParameterGroups(self, subset={}):
        for i in self.parameterGroups:
            lens = map(len,[self._parameterSubset(p,subset) for p in i])
            if not lens[:-1]==lens[1:]:
                raise(ValueError("Error in parameter group {}: not all have same number of entries.".format(i)))
    def _parameterSubset(self, parName,subset):
        if not parName in subset.keys(): return self.parameterValues[parName]
        else: return list(set(self.parameterValues[parName]).intersection(set(subset[parName])))
    def _parGroupToDict(self,g):
        return dict(g) if type(g[0]) is tuple else dict((g,))
    def _filterWithSubset(self,i,subset):
        parameters = self._parGroupToDict(i)
        for par in parameters.keys():
            if subset.get(par) and not parameters[par] in subset[par]: return False
        return True
    def parGen(self, subset={}):
        if len(self.parameterValues) == 0:
            yield dict()
            return
        def listify(l):
            return l if type(l[1]) is list else (l[0],[l[1]])
        subset = dict(map(listify,subset.items()))
        groupIterators = []
        singleParameters = self.parameterValues.keys()
        for group in self.parameterGroups:
            [singleParameters.remove(parameterName) for parameterName in group]
            zippedGroup = itertools.izip(*[[(parameterName,v) for v in self._parameterSubset(parameterName,subset)] for parameterName in group])
            groupIterators.append(itertools.ifilter(lambda i: self._filterWithSubset(i,subset),zippedGroup))
        for parameterName in singleParameters:
            groupIterators.append([(parameterName,v) for v in self._parameterSubset(parameterName,subset)])
        for combination in product(*groupIterators):
            thisCombination = {}
            for c in combination:
                thisCombination.update(self._parGroupToDict(c))
            yield thisCombination
