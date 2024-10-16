from lenetf9.arch.lenet.lenet_arch import LeNet
from lenetf9.bin.args.lenet_args import LenetArgs
from lenetf9.bin.modules.lenet import LeNetModules
from lenetf9.config.system import System


def main() -> None:
    args = LenetArgs.setup_args()
    model = LeNetModules(LeNet().to(System.DEVICE))
    if args and args.sub == "info":
        if args.info == "arch":
            model.model_summary()
        else:
            print("INFO COMMAND")
    elif args and args.sub == "train":
        parameters = {
            "cpu": args.cpu,
            "epochs": args.epochs,
            "step": args.step,
        }
        model.train(**{k: v for k, v in parameters.items() if v is not None})
    else:
        print("DEFAULT")


if __name__ == "__main__":
    main()
