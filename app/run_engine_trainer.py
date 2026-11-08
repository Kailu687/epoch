"""PyInstaller entry point. freeze_support() is required so spawned training workers work in a frozen build."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import os
    for name in ("stdout", "stderr"):  # windowed build has no console: keep print() from crashing
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    from engine_trainer.cli import main
    sys.exit(main())
