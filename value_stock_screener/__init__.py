"""价值选股独立模块。"""

def run_screener() -> None:
    from .app import main

    main()

__all__ = ["run_screener"]

