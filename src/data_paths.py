import os

# The spatial dataset jsons hold absolute paths from the machine they were generated on.
JSON_DATA_ROOT = "/storage/yotam/ssamba/data"


def remap_wav_path(path):
    """Rewrites a json wav path from JSON_DATA_ROOT to $SAC_DATA_ROOT (if set)."""
    data_root = os.environ.get("SAC_DATA_ROOT")
    if data_root and path.startswith(JSON_DATA_ROOT):
        return data_root.rstrip("/") + path[len(JSON_DATA_ROOT):]
    return path
