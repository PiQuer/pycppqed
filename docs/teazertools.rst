
======================
TeazerTools User Guide
======================

This is the documentation of additional tools which aim to simplify C++QED usage on the teazer cluster
of the Institute for Theoretical Physics (University of Innsbruck).

.. _submitter_documentation:

Submitter
=========

Summary
-------

The submitter framework aims to offer an easy to use interface to generic tasks of
sending one or several job arrays of C++QED trajectories to the teazer cluster and collecting the results. A configuration
file defines the name of the script to run, a set of seeds, the location where to save the results and all 
the relevant parameters for the script. From this information, a job array is created and 
submitted to the teazer cluster.  The job array simulates all the trajectories corresponding to the set of seeds.
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
* *matlab*: Convert output trajectories and statevectors to matlab format (default `True`)
* *numericsubdirs*: Instead of descriptive sub-directories which involve the values of the varied parameters,
  use numeric sub-directories 01/, 02/ etc. This can be convenient for further data procession. (default `False`)
* *combine*: If `True`, simulate all possible combinations of parameters with a range. If `False`, simulate
  one ensemble with the first value of all range parameters, one with the second value and so on until one
  of the ranges is exhausted (default `True`).
* *testrun_t*: Use this value as `-T` parameter in testruns (default 1)
* *testrun_dt*: Use this value as `-Dt` parameter in testruns (default: don't modify -Dt) 
  
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
parameter. The `value` can also be a range (a semicolon separated list of values). In this case, a trajectory ensemble
will be submitted for each value (or combination of values if several parameters have a range).

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
* ``--debug``: Very verbose debugging output.
* ``-h`` or ``--help``: Print help message.