
======================
TeazerTools User Guide
======================

This is the documentation of additional tools which aim to simplify C++QED usage on the teazer or leo3 cluster
of the University of Innsbruck.

.. _submitter_documentation:

Submitter
=========

Summary
-------

The submitter framework aims to offer an easy to use interface to generic tasks of
sending one or several job arrays of C++QED trajectories to an HPC cluster with sun grid engine (qsub) and 
collecting the results. A configuration
file defines the name of the script to run, a set of seeds, the location where to save the results and all 
the relevant parameters for the script. From this information, a job array is created and 
submitted to the HPC cluster.  The job array simulates all the trajectories corresponding to the set of seeds.
Optionally, an additional job is submitted, which averages the expectation 
values over all trajectories. This job depends on the job array, which means it is only executed when the last trajectory
has finished.

For the physical parameters, it is also possible do define a **range** in the configuration file, e.g.
`omegac=1;2;3;4`. If one or several parameters have such a range, a job array together with the averaging job 
is submitted for each possible combination of values within the ranges. In this case, the results are stored in
sub-directories named after the parameter combinations (or plain numbers, depending on configuration).

The behavior of the submitter is controlled by the class :class:`cppqedtools.GenericSubmitter`. This aims to be as
generic as possible and can be subclassed for more specific features, e.g. some parameters depending on some others. 

Example
-------

Take for example the following configuration file `example.conf`::

	[Config]
	script=1particle1mode
	basedir=/scratch/c705283/Data
	seeds=1001:1010
	
	[Averages]
	expvals=5;6
	variances=4;8
	stdevs=10
	
	[Parameters]
	dc=0
	Dt=0.1
	T=5
	deltaC=-10;-8
	kappa=5;6;7
	
This will generate the following directory structure::

	/scratch/c705283/Data/deltaC=-10_kappa=5/parameters.{txt,pkl,mat}
	                                        /log/...
	                                        /mean/...
	                                        /traj/...
	/scratch/c705283/Data/deltaC=-8_kappa=5/parameters.{txt,pkl,mat}
	                                       /log/...
	                                       /mean/...
	                                       /traj/...
	/scratch/c705283/Data/deltaC=-10_kappa=6/parameters.{txt,pkl,mat}
	                                        /log/...
	                                        /mean/...
	                                        /traj/...
	/scratch/c705283/Data/deltaC=-8_kappa=6/parameters.{txt,pkl,mat}
	                                       /log/...
	                                       /mean/...
	                                       /traj/...
	/scratch/c705283/Data/deltaC=-10_kappa=7/parameters.{txt,pkl,mat}
	                                       /log/...
	                                       /mean/...
	                                       /traj/...
	/scratch/c705283/Data/deltaC=-8_kappa=7/parameters.{txt,pkl,mat}
	                                       /log/...
	                                       /mean/...
	                                       /traj/...
	                                       
Each of the `traj` directories will contain the trajectories (C++QED output and matlab format) 
and the final statevectors. The `log` directories
will contain log messages and diagnostics of the jobs. The `mean` directories will each contain two files 
`1particle1mode_mean.npz` and `1particle1mode_mean.mat` with the averaged expectation values suitable to 
load into python and matlab, respectively. The files `parameters.{txt,pkl,mat}` contain all the parameters
with which the script was called in a human readable- python- and matlab format, respectively. These
parameters can be used in further data processing, e.g. plots.

Configuration File Syntax
-------------------------

The user has to provide a configuration file for the submitter on the command line. Values specified
here have highest priority. In addition, if the script to run is named `scriptname`, the file `~/.submitter/scriptname.conf`
is parsed if it exists. This is a convenient place to specify parameters and configuration values which are always
the same for a specific script. Finally, with lowest priority, the file `~/.submitter/generic_submitter.conf` is parsed if 
it exists.

The configuration files can contain consists of the following sections.

[Config]
________

The keywords in this section are (optional keywords italic)

* **script**: The C++QED binary to be called. This has to be in the `$PATH`, or an absolute path must
  be provided. The scriptname (with directory path stripped if provided) also serves as basename for
  trajectory and log files.
* **basedir**: The output data directory structure will be created relative to this path.
* **seeds**: This specifies the sets of seeds for each trajectory ensemble. This can be single seed number,
  a comma separated list of seeds or a range in matlab syntax (`start:step:stop`).
* *matlab*: (default `True`) Convert output trajectories and statevectors to matlab format.
* *average*: (default `True`) Calculate the averages of the expectation values (see :ref:`averages_ref`)
* *postprocess*: (default unset) name of a python class to perform more complex postprocessing of the data on the cluster 
  (see :ref:`postprocessing_ref`)
* *numericsubdirs*:  (default `True`) Instead of descriptive sub-directories which involve the values of the varied parameters,
  use numeric sub-directories 01/, 02/ etc. This can be convenient for further data procession. (default `False`)
* *testrun_t*: (default 1) Use this value as `-T` parameter in testruns.
* *testrun_dt*:  (default: don't modify -Dt) Use this value as `-Dt` parameter in testruns.
* *compress*:  (default `False`) Compress all trajectories and statevectors. Text files are compressed with bzip2, matlab files are
  compressed with matlabs own compression method. This can also serve as a backup in situations where no temporary
  directory is used: if a trajectory is continued, the compressed version of the trajectory file is kept until the 
  calculation was successful, only then is the compressed trajectory file updated.
* *resume*:  (default `False`) Use existing trajectories in the data directory to resume simulations. This is useful for two things: 1. to
  extend the integration to a larger value of T (existing trajectories are automatically copied to the temporary directory)
  2. to resume from failure: existing trajectories in the data directory which have the right final time T are untouched, 
  whereas missing trajectories are submitted again. Note that the averaging is always done over **all** trajectories in the
  data directory, the user has to make sure they have all the same length. Related options are `clean_seedlist`, `require_resume`
  and `continue_from`.  
* *clean_seedlist*: (default `True`) By default, before submitting the job array, all seeds are removed from the job array
  which have a final time that is already equal to `T`. This means if some trajectories fail one can just re-submit 
  everything and still only the failed trajectories will be simulated again.
* *require_resume*: (default `False`) If this is set to `True`, then a trajectory and the corresponding state vector file has to exist in the
  output directory, otherwise the seed is removed from the job array. The seed is also removed if compression is  activated
  but the an uncompressed output file is found. This is useful if one wants to continue some trajectories which are already 
  finished, whereas trajectories still in progress should not be touched.
* *continue_from*: (default: not set) If this is set to a time, then only those trajectories will be considered for resume
  which have this final time. One can use this switch if not all trajectories in the output directory are evolved to the same time.   
* *usetemp*: (default: `True`) Write the output file to a temporary directory on the node first, copy everything to
  scratch at the end. This is the preferred mode on teazer, whereas on leo3 this should be set to `False`.
* *cluster*: (default: 1) How many trajectories should be clustered into one job, each job simulates the trajectories one
  after the other. Use this to avoid scheduling overhead for very short trajectories.
* *parallel*: (default: 1) How many threads should be spawned. This can be combined with `cluster`. Note that each thread still
  uses a slot of the scheduler. This option can be used to request that always a complete node should be filled.
* *binary*: (default: `False`) Use binary output for state vector files. Note that C++Qed has to be built with the `enable-binary-output=yes`
  if this is set to `True`.   

.. _averages_ref:

[Averages]
__________
  
In order to calculate the averaged expectation values correctly, the script has to know which columns 
correspond to regular expectation values, variations and standard deviations, respectively. Column numbering
starts with 1.

Typically, the `[Averages]`-section will be in the file `~/.submitter/scriptname.conf`, as this is always the same for a script,
independent from the other parameters.

* *expvals*: Comma separated list of columns which contain expectation values.
* *variances*: Comma separated list of columns which contain variances.
* *stdevs*: Comma separated list of columns which contain standard deviations.

In order to calculate the averaged variances and standard deviations, the regular expectation values of
the observable has to be known. If not specified otherwise, this will be picked from the column 
just in front of the corresponding variance or standard deviation and need not to be specified in `expvals`.
In case this is not correct, the user can supply the additional keywords `varmeans` and `stdevmeans` with a
list of the correct positions.

[Parameters]
____________

Every `key=value` pair in this section will be passed on to the C++QED script as a ``--key value`` command line
parameter. The `value` can also be a range (a semicolon separated list of values or in Matlab syntax ``start:step:stop``). 
In this case, a trajectory ensemble will be submitted for each value (or combination of values if several parameters have a range).

You can refer to other values in the same configuration file by using `key=%(other_key)s`.

Parameters can be grouped. The values in each group are iterated ``in parallel``, that means there have to be the same
amount of values for each parameter within a group. Between different groups, all possible combinations of parameters are
generated. To define parameter groups, use `pargroupN=key1,key2,...`, where N is a number. For example, for a detuning scan
of a ring parameter with sine and cosine mode, one could use in the parameters section of the configuration file::

	[Parameters]
	pargroup1=deltaCSin,deltaCCos
	pargroup2=UnotSin,UnotCos
	deltaCSin=-8.5:0.5:-4.5
	deltaCCos=%(deltaCSin)s
	UnotSin=-2.5;-3.5
	UnotCos=%(UnotSin)s
	...

The definitions of the pargroups guarantee that the detuning and U0 is always the same for sine and the cosine mode in each run. 
The submitter will iterate over all detunings between -8.5 and -4.5 in steps of 0.5, and for each detuning it will use the two values of -2.5 and
-3.5 for U0. 

.. _postprocessing_ref :

Further postprocessing
______________________  

The method described in :ref:`averages_ref` is usually enough to calculate averages of the expectation values. However, sometimes
it is convenient to use the cluster for more involved postprocessing of the data. With the `postprocess` configuration parameter one
can name a python class. This class will be imported and an object will be generated on the executing node with the following signature:

	PostprocessingClass(basename, varPars, datapath, numericsubdirs)

* `basename` is the basename of all trajectory files.
* `varPars` is a :class:`teazertools.helpers.VariableParameters` object
* `datapath` is the path to the data, below which all the parameter set subdirectories reside
* `numericsubdirs` is a flag which indicates if the subdirectories are numeric or descriptive

The class has to implement a member function `postprocess` which takes an argument `subset`. This `subset` is a dictionary which
specifies a subset of all possible parameter values (see :meth:`teazertools.helpers.VariableParameters.parGen`). Upon successful creation of
the object, the member funciton `postprocess` is called.

Installation and Usage
----------------------

The users `$PYTHONPATH` has to include the package directory, e.g. 
`/home/c705283/pycppqed`, and the directory `pycppqed/bin` has to be added to
the `$PATH` so that the scripts can be found on the nodes. Calling the submitter is done by::

	   submitter [options] configfile

The options can be:

* ``--testrun``: The testrun flag will cause the submitter to use only two seeds for each ensemble and to 
  integrate up to `T=1` (if not set otherwise with the *testrun_t* option).
* ``--dryrun``: Don't actually submit anything to the teazer, instead print the commands that will be executed
  on the nodes (with the difference that the actual command will output data to a temporary directory first).
  This can be used to test if the command line is correct and the program will run properly.
* ``--class=CLASS``: Use CLASS as submitter class. This defaults to :class:`teazertools.submitter.GenericSubmitter`,
  and typically CLASS is a subclass of this to extend functionality.
* ``--verbose``: Verbose debugging output.
* ``--averageonly``: Only submit the job to compute the average expectation values (see :ref:`averages_ref`)
* ``--postprocessonly``: Only submit the job to do the advanced postprocessing (see :ref:`postprocessing_ref`)
* ``--class=CLASS``:  Use CLASS instead of :class:`teazertools.submitter.GenericSubmitter`,
  typically `CLASS` is a subclass of `GenericSubmitter`
* ``--depend=ID``:  Make created job array depend on this job ID.
* ``--subset=SUBSET``: a string describing a python dict to restrict all possible parameter values to the given subset, e.g. `{'par1':[1,2,3]}`
* ``-h`` or ``--help``: Print help message.