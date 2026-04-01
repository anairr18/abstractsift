import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _setup_dirs():
    import config
    for d in config.DIRS.values():
        Path(d).mkdir(parents=True, exist_ok=True)


def run_pipeline1():
    from pipeline1_patient_outcomes.pipeline1 import Pipeline1
    Pipeline1().run()


def run_pipeline2():
    from pipeline2_failed_trials.pipeline2 import Pipeline2
    Pipeline2().run()


def main():
    parser = argparse.ArgumentParser(description="Clinical ML Pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["1", "2", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    args = parser.parse_args()

    _setup_dirs()

    if args.pipeline in ("1", "both"):
        logger.info("Starting Pipeline 1: Patient Outcomes")
        run_pipeline1()

    if args.pipeline in ("2", "both"):
        logger.info("Starting Pipeline 2: Failed Trials")
        run_pipeline2()


if __name__ == "__main__":
    main()
