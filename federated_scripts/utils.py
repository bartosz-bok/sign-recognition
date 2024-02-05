import re


def extract_epoch_number(filename):
    """
    Extracts the epoch number from a model file name.

    :param filename: The name of the model file from which the epoch number is to be extracted.
    :return: The extracted epoch number from the file name, or 0 if the epoch number cannot be found.
    """
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


#
def print_pretty(models_dict) -> None:
    """
    Formatting and displaying the dictionary in a more readable way.

    :param models_dict: model dict to print
    """
    for model_name, files in models_dict.items():
        print(f"{model_name}:")
        files.sort()
        for file in files:
            print(f"  - {file}")
    print('\n')
