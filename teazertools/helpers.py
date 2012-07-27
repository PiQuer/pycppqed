""" This module defines some helper functions.
"""

import os
import errno
import pycppqed as qed
import warnings

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
    spec = map(int,s.split(':'))
    spec[-1] = spec[-1]+1
    if len(spec)==2:
        return range(*spec)
    elif len(spec)==3:
        return range(spec[0],spec[2],spec[1])
    else:
        raise ValueError('%s: range specification has to be start:[step:]stop'%s)
    
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
