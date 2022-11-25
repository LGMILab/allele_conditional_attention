import argparse
from pathlib import Path

# TODO: rewite this flags class to use in input file.
class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Immunogenicity prediction for neoantigen design.")
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument("--config-yml",
                                 required=True,
                                 type=Path,
                                 help="Path to a config file listing data, model, optim parameters.")
        self.parser.add_argument(
            "--run-dir",
            default="./",
            type=str,
            help="Directory to store checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--accelerator",
            choices=["train", "predict", "validate"],
            default='dp',
            help="Accelerator",
        )
        self.parser.add_argument(
            "--print-every",
            default=10,
            type=int,
            help="Log every N iterations (default: 10)",
        )
        self.parser.add_argument(
            "--seed", default=0, type=int, help="Seed for torch, cuda, numpy"
        )
        self.parser.add_argument(
            "--amp", action="store_true", help="Use mixed-precision training"
        )
        self.parser.add_argument(
            "--checkpoint", type=str, help="Model checkpoint to load"
        )
        # Cluster args
        self.parser.add_argument(
            "--sweep-yml",
            default=None,
            type=Path,
            help="Path to a config file with parameter sweeps",
        )

flags = Flags()