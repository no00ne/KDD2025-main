import os
import glob
import json
import pandas as pd

def compute_average_voyage_duration(json_dir):
    """
    Compute the average voyage duration (in hours) for all valid voyages
    stored as JSON or JSONL files under the given directory.

    Args:
        json_dir (str): Path to the directory containing .jsonl (or JSON array) files.

    Returns:
        float or None: The average duration in hours across all valid voyages.
                       Returns None if no valid voyage was found.
    """
    total_hours = 0.0
    count = 0

    # Find all .jsonl files under json_dir
    json_paths = glob.glob(os.path.join(json_dir, '*.jsonl'))

    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            # Peek first character to decide between JSON array vs JSONL
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # The file is a JSON array
                try:
                    voyages = json.load(f)
                except json.JSONDecodeError:
                    continue
                if not isinstance(voyages, list):
                    continue
            else:
                # The file is JSONL (one JSON object per line)
                voyages = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    voyages.append(obj)

        # Process each voyage record
        for voyage in voyages:
            # Attempt to parse Start Time and End Time
            try:
                ts_start = pd.to_datetime(voyage["Start Time"])
                ts_end = pd.to_datetime(voyage["End Time"])
            except Exception:
                continue

            # Only count if End > Start
            if ts_end <= ts_start:
                continue

            # Compute duration in hours
            delta_hours = (ts_end - ts_start).total_seconds() / 3600.0
            total_hours += delta_hours
            count += 1

    if count == 0:
        return None

    return total_hours / count


# Example usage:
# Suppose your JSONL files are in "./ship_trajectories"
if __name__ == "__main__":
    directory = "../ship_trajectories"
    avg_duration = compute_average_voyage_duration(directory)
    if avg_duration is None:
        print("No valid voyages found.")
    else:
        print(f"Average voyage duration: {avg_duration:.2f} hours")
