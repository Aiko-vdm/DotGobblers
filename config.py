from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = Path(PROJ_ROOT, "agents", "DotGobblers")


if __name__ == '__main__':
    print(f"The current project root is {PROJ_ROOT} \n")
    print(f"The current submission directory is {SUBMISSION_DIR}")
