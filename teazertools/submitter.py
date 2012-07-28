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
    :param config: A dictionary to control various aspects of this class.
    :type config: dict
    
    The possible values in `config` are:
    
    :param averageids: A dictionary to define which columns of the output files contain expectation values, variances and standard
        deviations.
    :type averageids: dict
    :param qsub: A dictionary of parameters passed to qsub. Each key:value pair corresponds to a
        -key value commandline option to qsub.
    :type qsub: dict
    :param qsub_traj: Same as `qsub`, but items here are only applied to qsub submissions of trajectory job arrays.
    :param qsub_average: Same as `qsub`, but items here are only applied to qsub submissions of averaging jobs.
    :type qsub_average: dict
    :param qsub_test: Same as `qsub`, but items here are only applied to qsub submission if it is a testrun.
    :param diagnostics: If `True`, print diagnostic messages to log files.
    :param matlab: If `True`, convert trajectories and state vector files to matlab format.
    :param average: If `True`, submit a job which calculates ensemble averages.
    :param usetemp: Write data to temporary directory first (default True).
    :param compress: Compress files (default True)
    :param resume: Resume trajectories (default False)
    :param testrun_t: Final time to integrate in testruns (default 1)
    :param testrn_dt: -Dt for testruns (default None)
    :param combine: If `True`, use all possible combinations of parameters (default True)
    :param cluster: Each job should calculate this many trajectories (default 1)
    """
    def __init__(self,script,basename=None,parameters={},basedir='.',tempdir='/tmp',seeds=[1001], config={}):
        self.C = dict(averageids={},qsub={}, qsub_traj={}, qsub_average={}, qsub_test={}, diagnostics=True,
                      matlab=True, average=True, compress=True, resume=False, testrun_t=1, testrun_dt = None,
                      usetemp=True, combine=True, cluster=1)
        self.C.update(config)
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
        self.compsuffix='.bz2'
        self.datafiles = []
        self.default_sub_pars = ['-b','y', '-v','PYTHONPATH','-v','PATH', '-m','n','-j','yes']
        self.loglevel = logging.getLogger().getEffectiveLevel()
        self.outputdir_is_temp = False
        self._warned =False
    
    def _prepare_exec(self,seed,dryrun):
        logging.debug("Entering _prepare_exec.")
        logging.debug(repr(self.parameters))
        self.parameters['seed'] = seed
        if self.C['usetemp']:
            self.outputdir = tempfile.mkdtemp(prefix=self.basename,dir=self.tempdir)
            self.outputdir_is_temp = True
        else:
            self.outputdir_is_temp = False
        self.command = [self.script]
        for item in self.parameters.items():
            self.command.extend(('--'+item[0],str(item[1])))
        self.targetoutput = self._targetoutput(**self.parameters)
        self.targetsv = self._targetsv(**self.parameters)
        self.output = helpers.replace_dirpart(self.targetoutput, self.outputdir)
        self.sv = helpers.replace_dirpart(self.targetsv, self.outputdir)
        if not dryrun:
            self.command.extend(('--o',self.output))
        
    def diagnostics_before(self):
        """This function is called before the executable is called. It can be overloaded in subclasses.
        The default implementation writes a message with the hostname to the log file.
        """
        logging.info("Output file is " + self.output)
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
        def _bzip2(f):
            os.system('bzip2 -f %s'%f)
        _bzip2(self.output)
        self.output = self.output + self.compsuffix
        if not self.C['binary']:
            _bzip2(self.sv)
            self.sv = self.sv + self.compsuffix
    
    def _move_data(self):
        for f in self.datafiles:
            shutil.copy(f, self.datadir)
            os.remove(f)
    
    def _cleanup(self):
        if self.outputdir_is_temp:
            logging.debug("Cleaning up on node, deleting %s."%self.outputdir)
            shutil.rmtree(self.outputdir, ignore_errors=True)
    
    def _convert_matlab(self):
        evs, svs = qed.load_cppqed(self.output)
        finalsv = qed.load_statevector(self.sv)
        scipy.io.savemat(self.output+".mat", {"evs":evs, "svs":svs}, do_compression=self.C['compress'])
        scipy.io.savemat(self.sv+".mat",{"sv":finalsv}, do_compression=self.C['compress'])
        self.datafiles.extend((self.output+".mat",self.sv+".mat"))
    
    def _targetoutput(self,seed=None,**kwargs):
        if seed:
            return self.targetoutputbase+'.'+str(seed)
        else:
            return self.targetoutputbase
    
    def _targetsv(self,seed=None,**kwargs):
        if self.C['binary']: ext='.svbin'
        else: ext='.sv' 
        return self._targetoutput(seed=seed)+ext
    
    def _find_target_files(self,seed=None,**kwargs):
        seed = str(seed)
        (targetoutput, output_compressed) = helpers.check_if_file_exists(self._targetoutput(seed),'.bz2')
        (targetsv, sv_compressed) = helpers.check_if_file_exists(self._targetsv(seed),'.bz2')
        return (targetoutput,output_compressed,targetsv,sv_compressed)
    
    
    def _keep_existing(self,seed):
        seed = str(seed)
        (targetoutput,output_compressed,targetsv,sv_compressed) = self._find_target_files(seed)
        if (not targetoutput or not targetsv) or (output_compressed != sv_compressed):
            if self.C['require_resume']:
                logging.info("Removing unfinished or nonexistent seed "+seed+".")
                return False
            else:
                logging.info("Keeping unfinished or nonexistent seed "+seed+".")
                return True
        if not self.parameters.has_key('T'):
            if not self._warned:
                logging.info("Please specify T. Note that CPPQed ignores T if NDt is given, but the submitter uses it to determine if a seed has to be included or not. Keeping all seeds.")
                self._warned=True
            return True
        lastT = helpers.cppqed_t(targetoutput)
        if lastT == None:
            logging.info('Could not read '+targetoutput+', keeping seed '+seed+'.') 
            return True
        T=float(self.parameters['T']) 
        if np.less_equal(T,float(lastT)):
            logging.info("Removing seed "+seed+ " from array, found trajectory with T=%f"%lastT)
            return False
        else:
            if self.parameters.has_key('NDt'):
                NDt=float(self.parameters['NDt'])
                Dt=float(self.parameters['Dt'])
                if not lastT+NDt*Dt==T:
                    logging.warn("Seed "+seed+ " with T=%f would not reach T=%f with NDt steps. Removing!"%(lastT,T))
                    return False
            else:
                if self.C.get('continue_from') and lastT != self.C.get('continue_from'):
                    logging.warn("Seed "+seed+" has T=%f, but %f required. Removing!"%(lastT,self.C.get('continue_from')))
                    return False
            logging.info("Keeping seed "+seed+ " with T=%f."%lastT)
            return True
        
    def _clean_seedlist(self):
        if not (self.C['resume'] and self.C['clean_seedlist']):
            return False
        logging.info("Checking for existing trajectories... this can take a long time")
        self.seeds[:] = [seed for seed in self.seeds if self._keep_existing(seed)]
    
    def _prepare_resume(self):
        """Puts everything in place to resume a trajectory.
        Returns True if nothing has to be simulated, returns False otherwise.
        """
        logging.debug("Entering _prepare_resume")
        (targetoutput,output_compressed,targetsv,sv_compressed) = self._find_target_files(**self.parameters)
        if not self.C['resume'] or not (targetoutput and targetsv):
            if targetoutput:
                logging.info("Deleting existing trajectory file %s."%targetoutput)
                os.remove(targetoutput)
            if targetsv:
                logging.info("Deleting existing sv file %s."%targetsv)
                os.remove(targetsv)
            return False
        
        lastT = helpers.cppqed_t(targetoutput)
        if lastT == None:
            logging.info("Found an invalid trajectory file %s."%targetoutput)
            return False
        logging.info("Found a trajectory with T=%f"%lastT)
        if self.parameters.has_key('T') and np.less_equal(float(self.parameters['T']),float(lastT)):
            logging.info("Don't need to calculate anything, T=%f."%float(self.parameters['T']))
            return True
        if self.C['usetemp']:
            logging.info('Moving %s to %s.'%(targetoutput,self.outputdir))
            shutil.copy(targetoutput, self.outputdir)
            targetoutput = helpers.replace_dirpart(targetoutput, self.outputdir)
            logging.debug('Moving %s to %s.'%(targetsv,self.outputdir))
            shutil.copy(targetsv, self.outputdir)
            targetsv = helpers.replace_dirpart(targetsv, self.outputdir)
        if output_compressed:
            logging.info('Uncompressing %s'%targetoutput)
            os.system('bunzip2 -k %s'%targetoutput)
        if sv_compressed:
            logging.info('Uncompressing %s'%targetsv)
            os.system('bunzip2 -k %s'%self.sv+self.compsuffix)
        return False
    
    def run(self, start=0, dryrun=False):
        """Simulate the trajectories self.seed[start:start+cluster] where cluster is the number of serial jobs. 
        
        :param start: The start value in the seed list.
        :type start: int
        :param dryrun: If `True`, don't simulate anything, but print a log message which contains the command that would
            have been run.
        """
        logging.debug("Entering run.")
        try:
            for s in self.seeds[start:start+self.C['cluster']]:
                self._prepare_exec(s,dryrun)
                if dryrun:
                    self._execute(self.command, dryrun, dryrunmessage="Executed on a node (with an additional appropriate -o flag):")
                    return
                if self._prepare_resume():
                    return
                if not os.path.exists(self.datadir): helpers.mkdir_p(self.datadir)
                self._write_parameters()
                if self.C['diagnostics']: self.diagnostics_before()
                (std,err,retcode) = self._execute(self.command)
                if not retcode == 0:
                    logging.error("C++QED script failed with exitcode %s:\n%s" % (retcode,err))
                    sys.exit(1)
                if self.C['diagnostics']: self.diagnostics_after()
                if self.C['matlab']:
                    self._convert_matlab()
                if self.C['compress']:
                    self._compress()
                self.datafiles.extend((self.output,self.sv))
            if self.C['usetemp']:
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
    
    def _dict_to_commandline(self,prefix,d):
        cl = []
        for i in d.items():
            if i[1]:
                for j in str(i[1]).split(';'):
                    cl.extend((prefix+i[0],j))
            else:
                cl.append(prefix+i[0])
        return cl
        
    def submit(self, testrun=False, dryrun=False):
        """Submit the job array to teazer. Technically this is done by serializing the object and passing it
        to the helper script `cppqedjob` as a commandline parameter. The helper script
        (running on a node) restores the object from the string and calls :func:`run`.
        
        :param testrun: Only simulate two seeds and set the parameter `T` to 1.
        :type testrun: bool
        """
        if not dryrun and testrun and (os.path.exists(self.datadir) or os.path.exists(self.averagedir)):
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
        numclusters = len(self.seeds)/self.C['cluster']+(len(self.seeds)%self.C['cluster']>0)
        numjobs = numclusters/self.C['parallel']+(numclusters%self.C['parallel']>0)
        if self.C['binary']: self.parameters['binarySVFile']=''
        if testrun:
            seedspec = "1-%s" % min(2,numjobs)
            self.parameters['T'] = self.C['testrun_t']
            if self.C['testrun_dt']: self.parameters['Dt'] = self.C['testrun_dt']
        else:
            seedspec = "1-%s" % numjobs
        
        obj = base64.encodestring(pickle.dumps(self,-1)).replace('\n','')
        logging.debug("String representation of JobArray object:")
        logging.debug(obj)
        
        command = ['qsub','-terse', '-o', logfile, '-N', jobname, '-t', seedspec]
        if self.C.get('depend'):
            command.extend(('-hold_jid',self.C['depend']))
        if self.C['parallel']>1: command.extend(('-pe','openmp',str(self.C['parallel'])))
        command.extend(self.default_sub_pars)
        command.extend(self._dict_to_commandline('-', self.C['qsub']))
        command.extend(self._dict_to_commandline('-', self.C['qsub_traj']))
        if testrun:
            command.extend(self._dict_to_commandline('-', self.C['qsub_test']))
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
        if self.C['average']:
            self.submit_average(holdid=jobid,dryrun=dryrun,testrun=testrun)
    
    def submit_average(self,holdid=None,dryrun=False,testrun=False):
        r"""Submit a job to teazer to compute the average expectation values.
        
        :param holdid: Make this job depend on the job array with id `holdid`.
        :type holdid: int
        :param dryrun: If `True`, don't submit anything, instead print the command that would
            have been called.
        :type dryrun: bool
        :returns returncode: qsub return value
        :retval: int
        """
        logfile = os.path.join(self.logdir,self.basename+'_mean_$JOB_ID.log')
        command = ['qsub','-terse', '-o', logfile, '-N', 'calculate_mean_{basedir}'.format(basedir=os.path.basename(self.basedir))]
        if holdid:
            command.extend(('-hold_jid',holdid))
        command.extend(self.default_sub_pars)
        command.extend(self._dict_to_commandline('-', self.C['qsub']))
        command.extend(self._dict_to_commandline('-', self.C['qsub_average']))
        if testrun:
            command.extend(self._dict_to_commandline('-', self.C['qsub_test']))
        command.append('calculate_mean')
        command.extend(self._dict_to_commandline('--', self.C['averageids']))
        command.extend(('--datadir',self.datadir))
        command.extend(('--outputdir',self.averagedir))
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
        self.JobArrayParams = {}
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
        self.combine = self.getboolean('Config', 'combine')
        self.numericsubdirs = self.getboolean('Config', 'numericsubdirs')
        self.basedir = self.JobArrayParams['basedir'] = os.path.expanduser(self.get('Config', 'basedir'))
        self.JobArrayParams['matlab'] = self.getboolean('Config', 'matlab')
        self.average = self.JobArrayParams['average'] = self.getboolean('Config', 'average')
        self.JobArrayParams['testrun_t'] = self.getfloat('Config', 'testrun_t')
        self.JobArrayParams['compress'] = self.getboolean('Config', 'compress')
        self.JobArrayParams['resume'] = self.getboolean('Config','resume')
        self.JobArrayParams['require_resume'] = self.getboolean('Config','require_resume')
        if self.has_option('Config', 'continue_from'):
            self.JobArrayParams['continue_from'] = self.getint('Config','continue_from')
        self.JobArrayParams['clean_seedlist'] = self.getboolean('Config', 'clean_seedlist')
        self.JobArrayParams['usetemp'] = self.getboolean('Config', 'usetemp')
        self.JobArrayParams['cluster'] = self.getint('Config', 'cluster')
        self.JobArrayParams['parallel'] = self.getint('Config', 'parallel')
        self.JobArrayParams['binary'] = self.getboolean('Config', 'binary')
        self.JobArrayParams['qsub'] = dict(self.items('Qsub'))
        self.JobArrayParams['qsub_traj'] = dict(self.items('QsubTraj'))
        self.JobArrayParams['qsub_average'] = dict(self.items('QsubAverage'))
        self.JobArrayParams['qsub_test'] = dict(self.items('QsubTest'))
        if ConfigParser.RawConfigParser.has_option(self,'Config', 'testrun_dt'):
            self.JobArrayParams['testrun_dt'] = self.getfloat('Config', 'testrun_dt')
        else:
            self.JobArrayParams['testrun_dt'] = None
        
        if self.JobArrayParams['average'] and self.has_section('Averages'):
            self.JobArrayParams['averageids'] = dict(self.items('Averages'))
        else: 
            logging.info('Averaging disabled: no information about output columns available. Please contact documentation about [Averages] section.')
            self.JobArrayParams['average'] = False

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
        self.add_option("--depend", dest="depend", metavar="ID",
                          help="Make created job array depend on this job ID.")
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
        if self.options.depend:
            self.JobArrayParams['depend'] = self.options.depend
        
        self.config = os.path.expanduser(args[0])
        
            
    def _jobarray_maker(self, basedir, parameters):
        myjob = JobArray(self.script, basedir=basedir, parameters=parameters, seeds=self.seeds[:], config=self.JobArrayParams)
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


