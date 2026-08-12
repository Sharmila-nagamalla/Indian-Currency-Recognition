import sys
import os


# Add project root to Python path

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


import splitfolders

from src.config import DATA_DIR



OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "split"
)



def split_dataset():

    print("Source Dataset:")
    print(DATA_DIR)


    print("\nOutput Dataset:")
    print(OUTPUT_DIR)


    splitfolders.ratio(
        DATA_DIR,
        output=OUTPUT_DIR,
        seed=42,
        ratio=(0.7, 0.2, 0.1)
    )


    print("\nDataset split completed successfully!")



if __name__ == "__main__":

    split_dataset()