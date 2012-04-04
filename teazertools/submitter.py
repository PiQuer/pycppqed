"""This module provides the infrastructure to send job arrays of C++QED trajectory ensembles to the teazer cluster.
"""

from optparse import OptionParser,OptionGroup
import ConfigParser
import os
import helpers
import logging
import itertools
import shutil
import cPickle as pickle
import scipy.io
import tempfile
import subprocess
import sys
import base64
import pycppqed as qed
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class JobArray(object):
    """This class represents a job array to simulate a trajectory ensemble. A job array is characterized by
    a set of seeds, one seed for each trajectory, and a set of C++QED parameters identical for all seed. Functions
    are provided to submit the job array to the teazer cluster (test run is possible) or to simulate a specific
    seed locally. For submitted job arrays, a further job depending on the array can be submitted to calculate
    the ensemble averages of expectation values.
    
    :param script: The name of the executable to call. This has to be an absolute path if it is not in the systems `$PATH`.
    :type script: str
    :param basename: The basename for output files. Defaults to the scriptname (without directory part).
    :type basename: str
    :param parameters: The C++QED parameters.
    :type parameters: dict
    :param basedir: The directory for the output directory structure.
    :type basedir: str
    :param tempdir: A temporary directory for C++QED output (submitted jobs write to local disk, the files are moved at the end)
    :type tempdir: str
    :param seeds: A list of seeds which defines the trajectory ensemble.
    :type seeds: list
    :param averageids: A dictionary to define which columns of the output files contain expectation values, variances and standard
        deviations.
    :type averageids: dict
    :param diagnostics: If `True`, print diagnostic messages to log files.
    :type diagnostics: bool
    :param matlab: If `True`, convert trajectories and state vector files to matlab format.
    :type matlab: bool
    :param average: If `True`, submit a job which calculates ensemble averages.
    :type average: bool
    """
    def __init__(self,script,basename=None,parameters={},basedir='.',tempdir='/tmp',seeds=[1001],
                 averageids={}, diagnostics=True, matlab=True, average=True):
        self.script = script
        self.parameters = parameters
        self.basedir=basedir
        self.datadir=os.path.join(basedir,'traj')
        self.outputdir=self.datadir
        self.logdir=os.path.join(basedir,'log')
        self.averagedir=os.path.join(basedir,'mean')
        self.tempdir=tempdir
        self.parameterfilebase=os.path.join(basedir,'parameters')
        if basename == None:
            self.basename = os.path.basename(script)
        else:
            self.basename = basename
        self.targetoutputbase=os.path.join(self.datadir,self.basename+'.out')
        self.seeds = seeds
        self.diagnostics = diagnostics
        self.teazer = False
        self.averageids = averageids
        self.matlab = matlab
        self.average = average
        self.compress = False
        self.compsuffix='.bz2'
        self.resume = False
        self.testrun_t = 1
        self.testrun_dt = None
        self.datafiles = []
        self.default_sub_pars = ['-b','y', '-v','PYTHONPATH','-v','PATH', '-q','all.q','-m','n','-j','yes']
        self.loglevel = logging.getLogger().getEffectiveLevel() 
    
    def _prepare_exec(self,seed,dryrun):
        logging.debug("Entering _prepare_exec.")
        if os.environ.has_key('SGE_TASK_ID'):
            self.teazer = True
            logging.debug("SGE_TASK_ID: "+os.environ['SGE_TASK_ID'])
            logging.debug(repr(self.parameters))
            self.parameters['seed'] = self.seeds[int(os.environ['SGE_TASK_ID'])-1]
            self.outputdir = tempfile.mkdtemp(prefix=self.basename,dir=self.tempdir)
        else:
            self.parameters['seed'] = self.seeds[seed]
        self.command = [self.script]
        for item in self.parameters.items():
            self.command.extend(('--'+item[0],str(item[1])))
        if self.parameters.has_key('seed'):
            suffix = '.%s'%self.parameters['seed']
        else:
            suffix = ''
        self.output = os.path.join(self.outputdir,self.basename+'.out'+suffix)
        self.sv = self.output+'.sv'
        self.targetoutput = self.targetoutputbase+suffix
        self.targetsv = self.targetoutput+'.sv'
        self.datafiles.extend((self.output, self.sv))
        if not dryrun:
            self.command.extend(('--o',self.output))
        
    def diagnostics_before(self):
        """This function is called before the executable is called. It can be overloaded in subclasses.
        The default implementation writes a message with the hostname to the log file.
        """
        logging.info("Job on %s started." % os.uname()[1])
    
    def diagnostics_after(self):
        """ This function is called after the executable finished. It can be overloaded in subclasses.
        The default implementation writes a message to the log files.
        """
        logging.info("Job finished.")
    
    def _numeric_parameters(self):
        numeric = {}
        for i in self.parameters.items():
            try:
                numeric[i[0]] = int(i[1])
                continue
            except: pass
            try:
                numeric[i[0]] = float(i[1])
                continue
            except: pass
            numeric[i[0]] = i[1]
        return numeric
    
    def _write_parameters(self):
        numeric = self._numeric_parameters()
        f = open(self.parameterfilebase+".txt",'w')
        f.write(repr(self.parameters)+"\n")
        f.close()
        f = open(self.parameterfilebase+".pkl",'w')
        pickle.dump(numeric, f, protocol=-1)
        f.close()
        scipy.io.savemat(self.parameterfilebase+".mat", numeric)
    
    def _compress(self):
        for f in self.datafiles:
            os.system('bzip2 %s'%f)
        self.datafiles = [f+self.compsuffix for f in self.datafiles]
    
    def _move_data(self):
        for f in self.datafiles:
            shutil.move(f, self.datadir)
    
    def _cleanup(self):
        if self.teazer:
            logging.debug("Cleaning up on node, deleting %s."%self.outputdir)
            shutil.rmtree(self.outputdir, ignore_errors=True)
    
    def _convert_matlab(self):
        if self.compress:
            suffix = '.bz2'
        else:
            suffix = ''
        evs, svs = qed.load_cppqed(self.output+suffix)
        finalsv = qed.load_statevector(self.sv+suffix)
        scipy.io.savemat(self.output+".mat", {"evs":evs, "svs":svs}, do_compression=self.compress)
        scipy.io.savemat(self.sv+".mat",{"sv":finalsv}, do_compression=self.compress)
        self.datafiles.extend((self.output+".mat",self.sv+".mat"))
    
    def _check_existing(self,seed):
        seed = str(seed)
        if os.path.exists(self.targetoutputbase+'.'+seed):
            target = self.targetoutputbase+'.'+seed
        elif os.path.exists(self.targetoutputbase+'.'+seed+self.compsuffix):
            target = self.targetoutputbase+'.'+seed+self.compsuffix
        else: return False
        lastT = helpers.cppqed_t(target)
        if lastT == None: return False
        if np.less_equal(float(self.parameters['T']),float(lastT)):
            logging.info("Removing seed "+seed+ " from array, found trajectory with T=%f",lastT)
            return True
        else:
            logging.info("Keeping seed "+seed+ " with T=%f.",lastT)
            return False
        
    def _clean_seedlist(self):
        if not self.resume:
            return False
        logging.info("Checking for existing trajectories... this can take a long time")
        self.seeds[:] = [seed for seed in self.seeds if not self._check_existing(seed)]
    
    def _prepare_resume(self):
        logging.debug("Entering _prepare_resume")
        if not self.resume:
            return False
        if os.path.exists(self.targetoutput):
            lastT = helpers.cppqed_t(self.targetoutput)
            target_traj_compressed = False
        elif os.path.exists(self.targetoutput+self.compsuffix):
            lastT = helpers.cppqed_t(self.targetoutput+self.compsuffix)
            target_traj_compressed = True
            self.targetoutput = self.targetoutput+self.compsuffix
        else:
            return False
        if lastT == None: return False
        logging.debug("Found a trajectory with T=%f"%lastT)
        if np.less_equal(float(self.parameters['T']),float(lastT)):
            logging.debug("Don't need to calculate anything, T=%f."%float(self.parameters['T']))
            return True
        if os.path.exists(self.targetsv):
            target_sv_compressed = False
        elif os.path.exists(self.targetsv+self.compsuffix):
            target_sv_compressed = True
            self.targetsv = self.targetsv+self.compsuffix
        if self.teazer:
            logging.debug('Moving %s to %s.'%(self.targetoutput,self.outputdir))
            shutil.copy(self.targetoutput, self.outputdir)
            logging.debug('Moving %s to %s.'%(self.targetsv,self.outputdir))
            shutil.copy(self.targetsv, self.outputdir)
        if target_traj_compressed:
            logging.debug('Uncompressing %s'%self.output+self.compsuffix)
            os.system('bunzip2 %s'%self.output+self.compsuffix)
        if target_sv_compressed:
            logging.debug('Uncompressing %s'%self.sv+self.compsuffix)
            os.system('bunzip2 %s'%self.sv+self.compsuffix)
        return False
    
    def run(self, seed=0, dryrun=False):
        """If the environment variable `$SGE_TASK_ID` is set (i.e. we are on a node), simulate the trajectory
        `self.seeds[$SGE_TASK_ID]`. Otherwise, simulate the trajectory `self.seeds[seed]` locally.
        
        :param seed: The seed to simulate locally if `$SGE_TASK_ID` is not set.
        :type seed: int
        :param dryrun: If `True`, don't simulate anything, but print a log message which contains the command that would
            have been run.
        """
        logging.debug("Entering run.")
        try:
            self._prepare_exec(seed,dryrun)
            if dryrun:
                self._execute(self.command, dryrun, dryrunmessage="Executed on a node (with an additional appropriate -o flag):")
                return
            if self._prepare_resume():
                return
            try:
                if not os.path.exists(self.datadir): helpers.mkdir_p(self.datadir)
            except OSError: pass
            self._write_parameters()
            if self.diagnostics: self.diagnostics_before()
            (std,err,retcode) = self._execute(self.command)
            if not retcode == 0:
                logging.error("C++QED script failed with exitcode %s:\n%s" % (retcode,err))
                sys.exit(1)
            if self.diagnostics: self.diagnostics_after()
            if self.compress:
                self._compress()
            if self.matlab:
                self._convert_matlab()
            if self.teazer:
                self._move_data()
        finally:
            self._cleanup()
            
    def _execute(self, command, dryrun=False, dryrunmessage="Would run command:", dryrunresult=("","")):
        logging.debug(subprocess.list2cmdline(command))
        if dryrun:
            logging.info(dryrunmessage + "\n" + subprocess.list2cmdline(command))
            (std,err) = dryrunresult
            returncode = 0
        else:
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (std,err) = p.communicate()
            returncode = p.returncode
        return (std,err,returncode)
        
    def submit(self, testrun=False, dryrun=False):
        """Submit the job array to teazer. Technically this is done by serializing the object, storing it in
        the environment variable $JobArray and calling the helper script `cppqedjob`. The helper script
        (running on a node) restores the object from the environment variable and calls :func:`run`.
        
        :param testrun: Only simulate two seeds and set the parameter `T` to 1.
        :type testrun: bool
        """
        if testrun and (os.path.exists(self.datadir) or os.path.exists(self.averagedir)):
            logging.error("The testrun potentially overwrites data in %s or %s. Will not start testrun while these directories exist."%(self.datadir,self.averagedir))
            sys.exit(1)
        if not dryrun:
            helpers.mkdir_p(self.logdir)
                
        jobname = "Job"+self.basename
        logfile = os.path.join(self.logdir,'$JOB_NAME.$JOB_ID.$TASK_ID.log')
        if not dryrun: self._clean_seedlist()
        if not self.seeds:
            logging.info('No seeds left to simulate.')
            return
        if testrun:
            seedspec = "1-%s" % min(2,len(self.seeds))
            self.parameters['T'] = self.testrun_t
            if self.testrun_dt: self.parameters['Dt'] = self.testrun_dt
        else:
            seedspec = "1-%s" % len(self.seeds)
        
        obj = base64.encodestring(pickle.dumps(self,-1)).replace('\n','')
        logging.debug("String representation of JobArray object:")
        logging.debug(obj)
        
        command = ['qsub','-terse','-v','JobArray', '-o', logfile, '-N', jobname, '-t', seedspec]
        command.extend(self.default_sub_pars)
        command.append('cppqedjob')
        if not dryrun:
            command.append(obj)
        (jobid,err,returncode) = self._execute(command, dryrun, dryrunresult=("100.0",""),
                                               dryrunmessage="Submit command on teazer:")
        if not returncode == 0:
            logging.error("Submit script failed.\n%s"%err)
            sys.exit(1)
        elif not dryrun:
            logging.info("Successfully submitted job id %s." %jobid.rstrip())
        jobid = jobid.split('.')[0]
        if self.average:
            self.submit_average(holdid=jobid,dryrun=dryrun)
    
    def submit_average(self,holdid=None,dryrun=False):
        r"""Submit a job to teazer to compute the average expectation values.
        
        :param holdid: Make this job depend on the job array with id `holdid`.
        :type holdid: int
        :param dryrun: If `True`, don't submit anything, instead print the command that would
            have been called.
        :type dryrun: bool
        :returns returncode: qsub return value
        :retval: int
        """
        logfile = os.path.join(self.logdir,self.basename+'_mean.log')
        command = ['qsub','-terse', '-o', logfile]
        if holdid:
            command.extend(('-hold_jid',holdid))
        command.extend(self.default_sub_pars)
        command.append('calculate_mean')
        for item in self.averageids.items():
            command.append('--'+item[0]+'='+item[1])
        command.append('--datadir='+self.datadir)
        command.append('--outputdir='+self.averagedir)
        command.append(self.basename)
        (jobid,err,returncode) = self._execute(command, dryrun, dryrunmessage="Submit command on teazer:")
        if returncode == 0:
            if not dryrun: logging.info("Submitted averaging script with job id %s."%jobid.rstrip())
        else:
            logging.error("Submit avarage failed.\n%s"%err)
            sys.exit(1)
        return returncode

class GenericSubmitter(OptionParser, ConfigParser.RawConfigParser):
    """ This class generates various :class:`JobArray` objects from a configuration file. For the syntax and usage, see
    :ref:`submitter_documentation`.
    """
    def __init__(self, argv=None):
        usage = "usage: %prog [options] configfile"
        ConfigParser.RawConfigParser.__init__(self)
        OptionParser.__init__(self,usage)
        if argv:
            sys.argv = argv
        self._parse_options()
    
        self.optionxform = str
        self.CppqedObjects = []
        self.defaultconfig = os.path.join(os.path.dirname(__file__),'generic_submitter_defaults.conf')
        self.averageids={}
        self._parse_config()
        self._generate_objects()
        
    def _parse_config(self):
        self.read(self.config)
        self.script = os.path.expanduser(self.get('Config','script'))
        self.read([self.defaultconfig,os.path.expanduser('~/.submitter/generic_submitter.conf'),
                     os.path.expanduser('~/.submitter/'+os.path.basename(self.script)+'.conf'), self.config])
        self.basedir = os.path.expanduser(self.get('Config', 'basedir'))
        self.matlab = self.getboolean('Config', 'matlab')
        self.average = self.getboolean('Config', 'average')
        self.numericsubdirs = self.getboolean('Config', 'numericsubdirs')
        self.combine = self.getboolean('Config', 'combine')
        self.testrun_t = self.getfloat('Config', 'testrun_t')
        self.compress = self.getboolean('Config', 'compress')
        self.resume = self.getboolean('Config','resume')
        if ConfigParser.RawConfigParser.has_option(self,'Config', 'testrun_dt'):
            self.testrun_dt = self.getfloat('Config', 'testrun_dt')
        else:
            self.testrun_dt = None
        
        if self.average and self.has_section('Averages'):
            self.averageids = dict(self.items('Averages'))
        else: self.average = False

        self.seeds = self.get('Config','seeds')
        if self.seeds.isdigit():
            self.seeds = [int(self.seeds)]
        elif self.seeds.count(';'):
            self.seeds = map(int,self.seeds.split(';'))
        elif self.seeds.count(':'):
            self.seeds = helpers.matlab_range_to_list(self.seeds)
        else:
            raise ValueError('Could not evaluate seeds specification %s.' %self.seeds)
    
    def _parse_options(self):
        self.add_option("--testrun", action="store_true", dest="testrun", default=False,
                          help="Submit test arrays to the cluster (only two seeds per ensemble, T=1 by default)")
        self.add_option("--dryrun", action="store_true", dest="dryrun", default=False,
                          help="Don't submit anything, only print out the commands that are executed on the nodes")
        self.add_option("--class", dest="classname", metavar="CLASS",
                          default='teazertools.submitter.GenericSubmitter',
                          help="Use CLASS instead of teazertools.submitter.GenericSubmitter, typically CLASS is a subclass of GenericSubmitter")
        self.add_option("--averageonly", action="store_true", dest="averageonly", default=False,
                          help="Only submit the job to compute the average expectation values")
        self.add_option("--verbose", action="store_true", dest="verbose", default=False,
                          help="Log more output to files.")
        
        group = OptionGroup(self, "Debugging options",
                        "These options are not needed for normal operation. "
                        "They provide means to debug the submitter.")
        group.add_option("--debug", action="store_true", help="Set breakpoint for external debugger.", default=False)
        group.add_option("--keeptemp", action="store_true", help="Don't delete temporary files. (Not implemented yet)")
        self.add_option_group(group)
        
        (self.options,args) = self.parse_args()
    
        if len(args) != 1:
            self.error("incorrect number of arguments")
        if self.options.verbose: logging.getLogger().setLevel(logging.DEBUG)
        if self.options.debug: 
            try:
                import pydevd 
                pydevd.settrace()
            except ImportError:
                logging.error("Pydevd module not found, cannot set breakpoint for external pydevd debugger.")
        
        self.config = os.path.expanduser(args[0])
        
            
    def _jobarray_maker(self, basedir, parameters):
        myjob = JobArray(self.script,basedir=basedir,seeds=self.seeds,averageids=self.averageids,
                         parameters=parameters, matlab=self.matlab, average=self.average)
        myjob.testrun_t = self.testrun_t
        myjob.testrun_dt = self.testrun_dt
        myjob.compress = self.compress
        myjob.resume = self.resume
        return myjob
        
    def _combine_pars(self, rangepars):
        if self.combine: generator = helpers.product(*rangepars)
        else: generator = itertools.izip(*rangepars)
        return generator
    
    def _generate_objects(self):
        pars = self.items('Parameters')
        singlepars = [i for i in pars if not i[1].count(';')]
        rangepars = [i for i in pars if i[1].count(';')]
        if not rangepars:
            self.CppqedObjects = [self._jobarray_maker(self.basedir,dict(pars))]
            return 
        expand = lambda x: [(x[0],i) for i in x[1].split(';') if not i == '']
        # expand: ('parname','val1,val2,val3') -> [('parname',val1),('parname',val2),('parname',val3)]
        rangepars = map(expand,rangepars)
        self.CppqedObjects = []
        generator = self._combine_pars(rangepars)
        counter = 1
        for parset in generator:
            localpars=dict(singlepars)
            localpars.update(dict(parset))
            if self.numericsubdirs:
                subdir = "%02d"%counter
            else: 
                subdir = '_'.join(["%s=%s"%i for i in parset])
            myjob = self._jobarray_maker(os.path.join(self.basedir,subdir), localpars)
            self.CppqedObjects.append(myjob)
            counter += 1
            
    def act(self):
        """Submit all job arrays to the teazer cluster.
        
        :param testrun: Perform a test run with only two seeds and `T=1`.
        :type testrun: bool
        :param dryrun: Don't submit anything, instead print what would be run on the nodes.
        :type dryrun: bool
        """
        
        for c in self.CppqedObjects:
            if self.options.averageonly:
                c.submit_average(dryrun=self.options.dryrun)
            else: 
                c.submit(testrun=self.options.testrun,dryrun=self.options.dryrun)
                if self.options.dryrun: c.run(dryrun=self.options.dryrun)


