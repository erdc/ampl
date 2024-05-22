import unittest
import json


class MyTestCase(unittest.TestCase):
    def test_something(self):
        filename = './results_Ballistics/journal_KER_test.json'

        with open(filename, 'r') as fp:
            full_journal = json.load(fp)
        self.assertEqual(full_journal['model']['target_variable'], 'KER')
        self.assertEqual(full_journal['model'].get('some_random_key'), None)
