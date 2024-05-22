import argparse

class PipelineCli(object):
    """
    Handle the parsing of commandline arguments. This will handle grabbing
    the yaml_file from commandline but may handle other commandline args
    in the future

    If other commandline arguments are needed, add them to the accepted
    Parser commands here
    """

    def __init__(self):

        self.pipeline = None
        self.config = None

        self.parser = argparse.ArgumentParser(prog='FAIT AMPL package',
                                              description='Process AMPL Pipeline Configuration YAML file.')

        self.parser.add_argument('config_file', type=str, help='Path to YAML Configuration file')
        self.parser.add_argument('-d', '--debug', action='store_true', help='Print debug info')
        self.parser.add_argument('-c', '--create_config', action='store_true',
                                 help='Creates a new default YAML Configuration file')
        self.parser.add_argument('-dt', '--decision_tree', action='store_true',
                                 help='Use Decision Tree instead of Neural Networks')
        self.parser.add_argument('-o', '--optuna', action='store_true',
                                       default=False, help='Run Optuna Model step')
        self.parser.add_argument('-b', '--build', action='store_true',
                                       default=False, help='Run Build Model step')
        self.parser.add_argument('-ev', '--evaluate', action='store_true',
                                       default=False, help='Run Evaluate Model step')
        self.parser.add_argument('-en', '--ensemble', action='store_true',
                                       default=False, help='Run Ensemble Model step')

        self.args = self.parser.parse_args()

    def main(self):
        args = self.args
        if args.create_config:
            from ampl import Util

            Util.create_default_config_file(args.config_file)
        else:
            from ampl.config import Configuration

            self.config = Configuration(args.config_file)

            if args.decision_tree:
                self.pipeline = self.config.create_pipeline_dt()
            else:
                self.pipeline = self.config.create_pipeline_nn()

            # if (self.config['data']['csv_file'] is not None and
            #         self.config['data']['csv_normalized_file'] is not None):
            #     self.pipeline.state.data.df.to_csv(self.config['data']['csv_normalized_file'], index=False, header=True)
            # elif (self.config['data']['data_table_name'] is not None and
            #       self.config['data']['data_normalized_table_name'] is not None):
            #     # write the preprocessed DataFrame to a SQLite database

                # from ampl.data import SqlUtil
                #
                # SqlUtil.to_sql(self.pipeline.state.data.df, self.config.create_data_db(),
                #                self.config['data']['data_normalized_table_name'])

            if args.optuna or args.build or args.evaluate or args.ensemble:
                if args.optuna:
                    self.pipeline.optuna.run()
                if args.build:
                    self.pipeline.build.run()
                if args.evaluate:
                    self.pipeline.eval.run()
                if args.ensemble:
                    self.pipeline.ensemble.run()
            else:
                self.pipeline.run_all()
