import json
import os

CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.zstack_anno_config.json')
DEFAULT_CONFIG = {
    'mask_folder': '',
    'script': []
}


def _load() -> dict:
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

_config = DEFAULT_CONFIG.copy()
_config.update(_load())


def save() -> None:
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(_config, f, indent=2)
    except Exception:
        pass


def get(key: str, default=None):
    return _config.get(key, default)


def set(key: str, value) -> None:
    _config[key] = value
