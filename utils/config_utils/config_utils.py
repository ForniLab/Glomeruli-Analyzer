from configparser import SafeConfigParser


class ConfigUtils:
    def __init__(self, config_path):
        super().__init__()
        self.parser = SafeConfigParser()
        self.parser.read(config_path)

    def get(self, section, option):
        return self.parser.get(section, option)

