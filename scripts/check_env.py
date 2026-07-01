import importlib.util
import sys


def is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")

    if is_installed("torch"):
        import torch

        print(f"torch installed: yes ({torch.__version__})")
        print(f"CUDA available: {torch.cuda.is_available()}")
    else:
        print("torch installed: no")
        print("CUDA available: unknown")

    if is_installed("ultralytics"):
        import ultralytics

        version = getattr(ultralytics, "__version__", "unknown")
        print(f"ultralytics installed: yes ({version})")
    else:
        print("ultralytics installed: no")


if __name__ == "__main__":
    main()
