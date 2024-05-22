import os
import sys
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from ampl.cli import PipelineCli

    logger.debug(os.getcwd())
    pipeline_cli = PipelineCli()

    sys.exit(pipeline_cli.main())
