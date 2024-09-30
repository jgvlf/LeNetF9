from lenetf9.bin.args.lenet_args import LenetArgs
from lenetf9.bin.modules.lenet import LeNetModules


def main() -> None:
    args = LenetArgs.setup_args()
    if args and args.sub == "info":
        if args.info == "arch":
            LeNetModules.model_summary()
        else:
            print("INFO COMMAND")
    elif args and args.sub == "train":
        parameters = {
            "cpu": args.cpu,
            "epochs": args.epochs,
            "step": args.step,
        }
        LeNetModules.train(**{k: v for k, v in parameters.items() if v is not None})
    else:
        print("DEFAULT")


if __name__ == "__main__":
    main()
