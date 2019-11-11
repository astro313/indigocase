import os

class Configurable(object):

    def __init__(self, config='../config.yaml', **kwargs):
        import yaml

        if os.path.isfile(config):
            self.config_dict = yaml.load(open(config))
        elif isinstance(config, dict) or config is None:
            self.config_dict = config
        elif not os.path.isfile(config):
            raise Exception("Invalid path to config file: %s." %config)
        else:
            raise Exception("Invalid config argument.")


if __name__ == '__main__':
    testC = Configurable()
    print(testC.config_dict)