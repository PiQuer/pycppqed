"""This module provides the infrastructure to send job arrays of C++QED trajectory ensembles to the teazer cluster.
"""


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
        self.seeds = seeds
        self.diagnostics = diagnostics
        self.teazer = False
        self.averageids = averageids
        self.matlab = matlab
        self.average = average
        
        self.testrun_t = 1
        self.testrun_dt = 0.1
        
        self.default_sub_pars = ['-b','y', '-v','PYTHONPATH','-v','PATH', '-q','all.q','-m','n','-j','yes']
    
    def _prepare_exec(self,seed,dryrun):
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
        if not dryrun:
            self.command.extend(('--o',self.output))
        
    def diagnostics_before(self):
        """This function is called before the executable is called. It can be overloaded in subclasses.
        The default implementation writes a message with the hostname to the log file.
        """
        logging.info("Job on %s started." % os.environ['HOSTNAME'])
    
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
        
    
    def _move_data(self):
        shutil.move(self.output, self.datadir)
        shutil.move(self.sv, self.datadir)
        if self.matlab:
            shutil.move(self.output+".mat", self.datadir)
            shutil.move(self.output+".sv.mat", self.datadir)
        shutil.rmtree(self.outputdir, ignore_errors=True)
    
    def _convert_matlab(self):
        evs, svs = qed.load_cppqed(self.output)
        finalsv = qed.load_statevector(self.output+".sv")
        scipy.io.savemat(self.output+".mat", {"evs":evs, "svs":svs})
        scipy.io.savemat(self.output+".sv.mat",{"sv":finalsv})
    
    def run(self, seed=0, dryrun=False):
        """If the environment variable `$SGE_TASK_ID` is set (i.e. we are on a node), simulate the trajectory
        `self.seeds[$SGE_TASK_ID]`. Otherwise, simulate the trajectory `self.seeds[seed]` locally.
        
        :param seed: The seed to simulate locally if `$SGE_TASK_ID` is not set.
        :type seed: int
        :param dryrun: If `True`, don't simulate anything, but print a log message which contains the command that would
            have been run.
        """
        self._prepare_exec(seed,dryrun)
        if dryrun:
            logging.info("This is the command executed on a node (with an additional appropriate -o flag):\n"
                         + subprocess.list2cmdline(self.command) + "\n")
            return
        if os.path.exists(self.output): os.remove(self.output)
        if os.path.exists(self.sv): os.remove(self.sv)
        try:
            if not os.path.exists(self.datadir): os.makedirs(self.datadir)
        except OSError: pass
        self._write_parameters()
        if self.diagnostics: self.diagnostics_before()
        retcode = subprocess.call(self.command)
        if not retcode == 0:
            logging.error("C++QED script failed with exitcode %s" % retcode)
            sys.exit(1)
        if self.diagnostics: self.diagnostics_after()
        if self.matlab:
            self._convert_matlab()
        if self.teazer:
            self._move_data()
 
    def submit(self, testrun=False):
        """Submit the job array to teazer. Technically this is done by serializing the object, storing it in
        the environment variable $JobArray and calling the helper script `cppqedjob`. The helper script
        (running on a node) restores the object from the environment variable and calls :func:`run`.
        
        :param testrun: Only simulate two seeds and set the parameter `T` to 1.
        :type testrun: bool
        """
        try:
            if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        except OSError: pass
        jobname = "Job"+self.basename
        logfile = os.path.join(self.logdir,'$JOB_NAME.$JOB_ID.$TASK_ID.log')
        if testrun:
            seedspec = "1-%s" % min(2,len(self.seeds))
            self.parameters['T'] = self.testrun_t
            self.parameters['Dt'] = self.testrun_dt
        else:
            seedspec = "1-%s" % len(self.seeds)
        
        os.environ['JobArray'] = base64.encodestring(pickle.dumps(self,-1)).replace('\n','')
        logging.debug("in submit:")
        logging.debug(os.environ['JobArray'])
        
        command = ['qsub','-terse','-v','JobArray', '-o', logfile, '-N', jobname, '-t', seedspec]
        command.extend(self.default_sub_pars)
        command.append('cppqedjob')
        logging.debug(repr(command))
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        (jobid,_) = p.communicate()
        if not p.returncode == 0:
            logging.error("Submit script failed.")
            sys.exit(1)
        else:
            logging.info("Successfully submitted job id %s." %jobid.rstrip())
        jobid = jobid.split('.')[0]
        if self.average:
            self._submit_average(jobid)
    
    def _submit_average(self,holdid):
        logfile = os.path.join(self.logdir,self.basename+'_mean.log')
        command = ['qsub','-o', logfile, '-hold_jid', holdid]
        command.extend(self.default_sub_pars)
        command.append('calculate_mean')
        for item in self.averageids.items():
            command.append('--'+item[0]+'='+item[1])
        command.append('--datadir='+self.datadir)
        command.append('--outputdir='+self.averagedir)
        command.append(self.basename)
        logging.debug(repr(command))
        retcode = subprocess.call(command)
        if retcode == 0:
            logging.info("Submitted averaging script.")
        else:
            logging.error("Submit avarage failed.")
            sys.exit(1)


class GenericSubmitter(object):
    """ This class generates various :class:`JobArray` objects from a configuration file. For the syntax and usage, see
    :ref:`submitter_documentation`.
    """
    def __init__(self, config):
        self.config = os.path.expanduser(config)
    
        self.c = ConfigParser.RawConfigParser()
        self.c.optionxform = str
        self.CppqedObjects = []
        self.defaultconfig = os.path.join(os.path.dirname(__file__),'generic_submitter_defaults.conf')
        self.averageids={}
        self._parse_config()
        self._generate_objects()
        
    def _parse_config(self):
        self.c.read(self.config)
        self.script = os.path.expanduser(self.c.get('Config','script'))
        self.c.read([self.defaultconfig,os.path.expanduser('~/.submitter/generic_submitter.conf'),
                     os.path.expanduser('~/.submitter/'+os.path.basename(self.script)+'.conf'), self.config])
        self.basedir = os.path.expanduser(self.c.get('Config', 'basedir'))
        self.matlab = self.c.getboolean('Config', 'matlab')
        self.average = self.c.getboolean('Config', 'average')
        self.numericsubdirs = self.c.getboolean('Config', 'numericsubdirs')
        self.combine = self.c.getboolean('Config', 'combine')
        self.testrun_t = self.c.getfloat('Config', 'testrun_t')
        self.testrun_dt = self.c.getfloat('Config', 'testrun_dt')
        
        if self.average and self.c.has_section('Averages'):
            self.averageids = dict(self.c.items('Averages'))
        else: self.average = False

        self.seeds = self.c.get('Config','seeds')
        if self.seeds.isdigit():
            self.seeds = [int(self.seeds)]
        elif self.seeds.count(';'):
            self.seeds = map(int,self.seeds.split(';'))
        elif self.seeds.count(':'):
            self.seeds = helpers.matlab_range_to_list(self.seeds)
        else:
            raise ValueError('Could not evaluate seeds specification %s.' %self.seeds)
        
            
    def _jobarray_maker(self, basedir, parameters):
        myjob = JobArray(self.script,basedir=basedir,seeds=self.seeds,averageids=self.averageids,
                         parameters=parameters, matlab=self.matlab, average=self.average)
        myjob.testrun_t = self.testrun_t
        myjob.testrun_dt = self.testrun_dt
        return myjob
        
    
    def _generate_objects(self):
        pars = self.c.items('Parameters')
        singlepars = [i for i in pars if not i[1].count(';')]
        rangepars = [i for i in pars if i[1].count(';')]
        if not rangepars:
            self.CppqedObjects = [self._jobarray_maker(self.basedir,dict(pars))]
            return 
        expand = lambda x: [(x[0],i) for i in x[1].split(';')]
        # expand: ('parname','val1,val2,val3') -> [('parname',val1),('parname',val2),('parname',val3)]
        rangepars = map(expand,rangepars)
        self.CppqedObjects = []
        counter = 1
        if self.combine: generator = helpers.product(*rangepars)
        else: generator = itertools.izip(*rangepars)
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
            
    def submit(self, testrun=False, dryrun=False):
        """Submit all job arrays to the teazer cluster.
        
        :param testrun: Perform a test run with only two seeds and `T=1`.
        :type testrun: bool
        :param dryrun: Don't submit anything, instead print what would be run on the nodes.
        :type dryrun: bool
        """
        for c in self.CppqedObjects:
            if dryrun: c.run(dryrun=True)
            else: c.submit(testrun)
