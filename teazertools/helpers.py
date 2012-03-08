""" This module defines some helper functions.
"""

import os
import errno

def mkdir_p(path):
    """http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise


def generate_filelist(basename,dirname):
    """ This function generates a list of files which start with `basename` and either end with a digit (possibly with an `.bz2` extension)
    
    :param basename: The files have to begin with this string.
    :type basename: str
    :param dirname: Search in this directory.
    :type dirname: str
    :returns: A list of matching filenames.
    :retval: list
    """
    filelist = [ os.path.join(dirname,x) for x in os.listdir(dirname) if x.startswith(basename) and 
                        ((not x.endswith('.bz2') and x[-1].isdigit()) or (x.endswith('.bz2') and x[-5].isdigit() ))]
    assert filelist != []
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