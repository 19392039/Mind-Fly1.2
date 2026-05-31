from tensorboard import program


def main() -> int:
    tb = program.TensorBoard()
    tb.configure(
        argv=[
            "tensorboard",
            "--logdir",
            r"C:\Users\Administrator\Desktop\data",
            "--host",
            "127.0.0.1",
            "--port",
            "6006",
        ]
    )
    return tb.main()


if __name__ == "__main__":
    raise SystemExit(main())
