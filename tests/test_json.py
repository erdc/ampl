import unittest
import json


class MyTestCase(unittest.TestCase):
    def test_something(self):
        filename = './results_study1/journal_y_test.json'

        with open(filename, 'r') as fp:
            full_journal = json.load(fp)
        self.assertEqual(full_journal['model']['target_variable'], 'y')
        self.assertEqual(full_journal['model'].get('some_random_key'), None)
