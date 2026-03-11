

def get_csv_path(csv_name: str) -> str:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, csv_name)
    return csv_path

def get_directory_name(instruction: str) -> str:
    instruction = instruction.lower().replace(" ", "_").replace(".", "")
    return instruction
