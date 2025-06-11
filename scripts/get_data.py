
"""
get_data.py
===========

Prepare cached NPZ files (probabilities + fold indices,etc) for any task family.



Usage examples
--------------
# build NPZs for *all* MCC datasets
python -m scripts.get_data --task MCC

or for a specific dataset
python -m scripts.get_data --task MCC --datasets glass


"""
from pathlib import Path
import argparse
import sys


# 1. add project root to PYTHONPATH so we can import data.prepare_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.prepare_data import prepare_all  

# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NPZ caches of probabilistic predictions "
                    "for MCC, MLC or MDC datasets."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("MCC", "MLC", "MDC"),
        help="Task family whose datasets you want to process",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        metavar="NAME",
        help="Dataset *stems* to process (omit .csv). "
             "If omitted, all CSVs in data/<TASK>/ are used.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        metavar="DIR",
        help="Top-level data directory that contains MCC/, MLC/, MDC/ (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default="cache",
        metavar="DIR",
        help="Where NPZ files will be written (default: %(default)s)",
    )
    args = parser.parse_args()

    
    # build path to the task-specific subfolder, e.g. data/MCC
    
    subdir = Path(args.data_root) / args.task.upper()
    if not subdir.exists():
        parser.error(f"Folder {subdir} does not exist")

   
    # collect dataset names (file stems) if the user didn't specify
    
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = [p.stem for p in subdir.glob("*.csv")]
        if not dataset_names:
            parser.error(f"No .csv files found in {subdir}")

    # --------------------------------------------------------------
    # run the heavy lifting (prepare_all lives in data/prepare_data.py)
    
    prepare_all(
        task_type=args.task,
        dataset_names=dataset_names,
        data_dir=subdir,          # pass *task folder* to prepare_all
        out_dir=args.out_dir,
    )

    print("\nDone.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
