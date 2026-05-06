import yaml
from .constructors import RandomLoader, DeterministicLoader
from .dumpers import TaggedDumper
from .tagged_types import Tagged


class SettingParser:
    # __sampled_setting__ holds the raw YAML (with !rand* resolved) separately
    # from the instantiated objects in __dict__, so to_dict() exposes only the latter.
    __slots__ = ['__dict__', '__sampled_setting__']

    def __init__(self, setting_path: str) -> None:
        self.__sampled_setting__ = self.sample_setting(setting_path)

    @staticmethod
    def sample_setting(fname: str):
        """First pass: resolve !randint / !randfloat / !randlog10 / !choice."""
        with open(fname, 'r') as file:
            return yaml.load(file, Loader=RandomLoader)

    @staticmethod
    def dump_settings(sampled):
        return yaml.dump(sampled, Dumper=TaggedDumper,
                         default_flow_style=False, allow_unicode=True)

    def load_setting(self):
        """Second pass: instantiate !model / !dataset / !optimizer / !metric / !activation."""
        dumped_setting = self.dump_settings(self.__sampled_setting__)
        self.__dict__ = yaml.load(dumped_setting, Loader=DeterministicLoader)
        return self

    def update_setting(self, new_setting: str):
        """Merge a YAML-format string into the sampled setting (e.g. `device: cuda:0`)."""
        update_setting = yaml.load(new_setting, Loader=RandomLoader)
        self.__sampled_setting__.update(update_setting)

    def save_setting(self, setting_path: str):
        with open(setting_path, 'w') as file:
            file.write(self.dump_settings(self.__sampled_setting__))

    def __item__(self, key: str):
        """Recursive lookup: return the first value bound to `key` at any nesting level."""
        def _finditem(key: str, obj: dict):
            if key in obj.keys():
                value = obj[key]
                if isinstance(value, Tagged):
                    value = value.value
                return value
            for v in obj.values():
                value = v
                if isinstance(v, Tagged):
                    value = v.value
                if isinstance(value, dict):
                    item = _finditem(key, value)
                    if item is not None:
                        return item
        return _finditem(key, self.__sampled_setting__)

    def __repr__(self) -> str:
        return self.dump_settings(self.__sampled_setting__)

    def to_dict(self):
        return vars(self)
