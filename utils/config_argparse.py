import argparse
from pathlib import Path

import yaml


class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, ignore_unknown_config_keys: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_unknown_config_keys = ignore_unknown_config_keys
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not isinstance(d, dict):
                self.error("Config file has non dict value: {_args.config}")

            known_defaults = {}
            for key, value in d.items():
                for action in self._actions:
                    if key == action.dest:
                        known_defaults[key] = value
                        break
                else:
                    if not self.ignore_unknown_config_keys:
                        self.error(f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't check type.
            #   i.e. We can set any type value regardless of argument type.
            self.set_defaults(**known_defaults)
        return super().parse_known_args(args, namespace)
