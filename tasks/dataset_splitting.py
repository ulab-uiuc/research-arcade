import sys
import os
import csv
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation import _data_fetching

def data_splitting(
    csv_id_path,
    data_fetching_method,
    output_dir="./data/paragraph_generation",
    file_name_prefix="paragraph_generation",
    training=0.7,
    validation=0.1,
    testing=0.2,
    seed=42,
    save_split_id_lists=True,   # set False if you don’t want sidecar files
):
    """
    Splits paragraph IDs into training/validation/testing sets
    and fetches the data using `data_fetching_method`.

    Args:
        csv_id_path (str): Path to the CSV file with 'id' column.
        data_fetching_method (function): Function that takes
            (paragraph_key_ids, data_path) and writes JSONL.
        output_dir (str): Directory to save splits.
        file_name_prefix (str): Prefix for output files.
        training, validation, testing (float): Split ratios.
        seed (int): Random seed for reproducibility.
        save_split_id_lists (bool): Save split IDs for auditability.
    """
    # Basic sanity
    assert abs((training + validation + testing) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Collect paragraph IDs
    paragraph_key_ids = []
    with open(csv_id_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',')
        for row in csv_reader:
            # Defensive: skip empty/malformed rows
            if not row or 'id' not in row or not row['id']:
                continue
            paragraph_key_ids.append(row['id'].strip())

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(paragraph_key_ids)

    # Split counts
    n_row = len(paragraph_key_ids)
    n_training = int(training * n_row)
    n_validation = int(validation * n_row)
    n_testing = n_row - n_training - n_validation  # absorb rounding

    # Partition
    training_ids   = paragraph_key_ids[:n_training]
    validation_ids = paragraph_key_ids[n_training:n_training + n_validation]
    testing_ids    = paragraph_key_ids[n_training + n_validation:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    training_data_path   = os.path.join(output_dir, f"{file_name_prefix}_training.jsonl")
    validation_data_path = os.path.join(output_dir, f"{file_name_prefix}_validation.jsonl")
    testing_data_path    = os.path.join(output_dir, f"{file_name_prefix}_testing.jsonl")

    # Fetch & write JSONL using the provided method
    data_fetching_method(paragraph_key_ids=training_ids,   data_path=training_data_path)
    data_fetching_method(paragraph_key_ids=validation_ids, data_path=validation_data_path)
    data_fetching_method(paragraph_key_ids=testing_ids,    data_path=testing_data_path)

    # (Optional) Save the split ID lists
    if save_split_id_lists:
        for name, ids in [
            ("training_ids.txt", training_ids),
            ("validation_ids.txt", validation_ids),
            ("testing_ids.txt", testing_ids),
        ]:
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(ids))

    print(f"Split complete: {n_training} train, {n_validation} val, {n_testing} test.")
    print(f"Wrote:\n  {training_data_path}\n  {validation_data_path}\n  {testing_data_path}")

def main():
    data_splitting(
        csv_id_path="./csv/paragraphs_with_figures_tables_csv.csv",
        output_dir="./data/paragraph_generation",
        data_fetching_method=_data_fetching,
    )

if __name__ == "__main__":
    main()
