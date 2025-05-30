import json
import os

def split_json_file(input_path, output_dir):
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Split data based on 'is_unpredictable' and remove this attribute
    predictable = []
    unpredictable = []

    for entry in data:
        is_unpredictable = entry.pop("is_unpredictable", False)
        if is_unpredictable:
            unpredictable.append(entry)
        else:
            predictable.append(entry)

    # Write results into two separate files
    with open(os.path.join(output_dir, 'predictable.json'), 'w', encoding='utf-8') as file:
        json.dump(predictable, file, indent=4)

    with open(os.path.join(output_dir, 'unpredictable.json'), 'w', encoding='utf-8') as file:
        json.dump(unpredictable, file, indent=4)

    print("Splitting completed successfully.")

# Example usage
split_json_file('./result-v1/evaluations-R1-sorted.json', '../result_data')
