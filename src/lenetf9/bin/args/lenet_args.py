from argparse import Namespace


class LenetArgs:
    @staticmethod
    def setup_args() -> Namespace:
        import argparse

        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            usage="lenet [OPTIONS] <COMMAND>",
        )
        subparsers = parser.add_subparsers(title="Commands", metavar="", dest="sub")

        info = subparsers.add_parser("info", help="LeNet informations", usage="lenet info [OPTIONS] <COMMAND>")
        sub_info = info.add_subparsers(title="Commands", metavar="", dest="info")

        arch = sub_info.add_parser(
            "arch",
            help="Information of Neural Network Arch",
            usage="lenet info arch [OPTIONS]",
        )
        arch_parameters = arch.add_argument_group("Parameters")
        arch_parameters.add_argument("--author", action="store_true")

        subparsers.add_parser("train", help="Training LeNet model")
        return parser.parse_args()
