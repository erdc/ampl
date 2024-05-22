from unittest import TestCase

from ampl.util import Util
import os


class TestUtil(TestCase):
    def test_get_n_jobs(self):
        self.assertEqual(-1, Util.get_n_jobs(percent=0.1))
        self.assertEqual(-1, Util.get_n_jobs(percent=0.09))
        self.assertEqual(-1, Util.get_n_jobs(percent=0.91))

        per = 0.9
        self.assertEqual(int(os.cpu_count()*per), Util.get_n_jobs(percent=per))

        per = 0.5
        self.assertEqual(int(os.cpu_count()*per), Util.get_n_jobs(percent=per))
