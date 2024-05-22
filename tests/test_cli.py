from unittest import TestCase
from unittest.mock import patch

from ampl.cli import PipelineCli


class TestPipelineCli(TestCase):

    @patch('sys.argv', ['mock.py', '-h'])
    def test_main_00(self):
        import sys
        with self.assertRaises(SystemExit) as cm:
            pipeline_cli = PipelineCli()

        self.assertEqual(cm.exception.code, 0)


    @patch('sys.argv', ['mock.py', 'data/pipeline_config.yml'])
    def test_main_0(self):
        pipeline_cli = PipelineCli()
        pipeline_cli.main()

    # testing decision tree option
    @patch('sys.argv', ['mock.py', 'data/pipeline_config.yml', '-dt'])
    def test_main_1(self):
        pipeline_cli = PipelineCli()
        pipeline_cli.main()

    # testing build option
    @patch('sys.argv', ['mock.py', 'data/pipeline_config.yml', '-b'])
    def test_main_2(self):
        pipeline_cli = PipelineCli()
        pipeline_cli.main()

    @patch('sys.argv', ['mock.py', 'data/pipeline_config.yml', '-dt', '-b'])
    def test_main_3(self):
        pipeline_cli = PipelineCli()
        pipeline_cli.main()
