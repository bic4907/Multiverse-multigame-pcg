import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def load_dotenv(override: bool = True) -> bool:
    """
    Load environment variables from a .env file (manual implementation).

    Args:
        env_path: Path to .env file
        override: Whether to override existing environment variables

    Returns:
        bool: True if loaded, False if file does not exist
    """
    if not os.path.exists(ROOT_DIR):
        return False

    with open(ROOT_DIR, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty lines & comments
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue  # invalid line, ignore

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # remove surrounding quotes
            if (
                (value.startswith('"') and value.endswith('"')) or
                (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value

    return True

if __name__ == "__main__":
    # Example usage
    environ = load_dotenv()
