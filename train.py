from __future__ import annotations

from trainer import train_from_config


def main() -> None:
    train_from_config("config.json")


if __name__ == "__main__":
    main()

