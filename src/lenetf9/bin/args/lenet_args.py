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

        sub_info.add_parser(
            "arch",
            help="Information of Neural Network Arch",
            usage="lenet info arch [OPTIONS]",
        )
        train = subparsers.add_parser("train", help="Training LeNet model", usage="lenet train [OPTIONS] <COMMAND>")
        train_args = train.add_argument_group("Parameters")
        train_args.add_argument("--cpu", action="store_true", help="Train with the CPU")
        train_args.add_argument("--epochs", type=int, help="Define the number of training cycles")
        train_args.add_argument("--step", type=int, help="Define de distance between the training prints")

        return parser.parse_args()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
