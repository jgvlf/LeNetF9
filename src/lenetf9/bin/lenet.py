from lenetf9.bin.args.lenet_args import LenetArgs


def main() -> None:
    args = LenetArgs.setup_args()
    if args and args.sub == "info":
        if args.info == "arch":
            print("ARCH COMMAND INSIDE INFO")
        else:
            print("INFO COMMAND")
    elif args and args.sub == "train":
        print("TRAIN!!!")
    else:
        print("DEFAULT")


if __name__ == "__main__":
    main()
