import unittest
import mean
import numpy as np
import submitter
from mock import *
import subprocess
import cPickle as pickle
import pycppqed as qed


class TestMean(unittest.TestCase):

    def test01_calculateMeans1(self):
        result = mean.calculateMeans("test1", expvals=[], variances=[4], stdevs=[6], datadir='test/',usesaved=False,outputdir=None)
        self.assertTrue(np.allclose(result,np.load('test/test1expected.npy')))
    
    def test02_calculateMeans2(self):
        result = mean.calculateMeans("test2", expvals=[3,5], variances=[4], stdevs=[6], datadir='test/',usesaved=False,outputdir=None)
        self.assertTrue(np.allclose(result,np.load('test/test2expected.npy')))


class TestSubmitter(unittest.TestCase):
    
    def setUp(self):
        self.popen = Mock()
        self.popen.returncode = 0
        self.popen.communicate = Mock(return_value=('574599.1-2:1',''))
        self.popen_patch = patch('subprocess.Popen',new=Mock(return_value=self.popen))
        self.call_patch = patch('subprocess.call',new=Mock(return_value=0))
        self.expected_args_regular = []
        self.expected_args_averaging = []
        for i in range(4):
            self.expected_args_regular.append(['qsub', '-terse', '-v', 'JobArray', '-o', 'test/output/%02d/log/$JOB_NAME.$JOB_ID.$TASK_ID.log'%(i+1), 
                                            '-N', 'Job1particle1mode', '-t', '1-10', '-b', 'y', '-v', 'PYTHONPATH', '-v', 'PATH', '-q', 'all.q', 
                                            '-m', 'n', '-j', 'yes', 'cppqedjob'],)
            self.expected_args_averaging.append(['qsub', '-terse','-o', 'test/output/%02d/log/1particle1mode_mean.log'%(i+1), '-hold_jid', '574599', '-b', 'y', '-v', 'PYTHONPATH', 
                                           '-v', 'PATH', '-q', 'all.q', '-m', 'n', '-j', 'yes', 'calculate_mean', '--variances=4,8', '--expvals=5,6', 
                                           '--stdevs=10', '--datadir=test/output/%02d/traj'%(i+1), '--outputdir=test/output/%02d/mean'%(i+1), '1particle1mode'],)
        
    def test01_submit(self):
        s = submitter.GenericSubmitter('test/test.conf')
        with self.popen_patch as p:
            s.submit()
            for i in range(4):
                self.assertEqual(((self.expected_args_regular[i],),{'stdout':subprocess.PIPE,'stderr':subprocess.PIPE}),p.call_args_list[2*i])
                self.assertEqual(((self.expected_args_averaging[i],),{'stdout':subprocess.PIPE,'stderr':subprocess.PIPE}),p.call_args_list[2*i+1])
    
    def test02_run(self):
        s = submitter.GenericSubmitter('test/test.conf')
        s.CppqedObjects[0].run()
        f = open('test/output/01/parameters.pkl')
        pars = pickle.load(f)
        f.close()
        self.assertEqual(pars, {'kappa': 0.1, 'dc': 0, 'deltaC': -8, 'seed': 1001, 'T': 0.5, 'Dt': 0.1})
        qed.load_cppqed('test/output/01/traj/1particle1mode.out.1001')
        qed.load_statevector('test/output/01/traj/1particle1mode.out.1001.sv')
        

                    
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()