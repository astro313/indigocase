from indigocase.src.parser import Parser
from indigocase.src.config import Configurable
import os


def test_yaml_input_fields():

    ccc = Configurable()
    setupParam = ccc.config_dict

    dpath = setupParam['input']['dataPath']
    trainfile = os.path.join(dpath, setupParam['input']['trainFile'])
    testfile = os.path.join(dpath, setupParam['input']['testFile'])
    truthfile = os.path.join(dpath, setupParam['input']['truthFile'])

    assert os.path.isfile(trainfile)
    assert os.path.isfile(testfile)
    assert os.path.isfile(truthfile)

