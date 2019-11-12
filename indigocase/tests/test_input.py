from indigocase.src.parser import Parser
from indigocase.src.config import Configurable
import os


def test_yaml_input_fields(config='indigocase/config.yaml'):

    ccc = Configurable(config=config)
    setupParam = ccc.config_dict

    dpath = setupParam['input']['dataPath']
    trainfile = os.path.join(dpath, setupParam['input']['trainFile'])
    testfile = os.path.join(dpath, setupParam['input']['testFile'])
    truthfile = os.path.join(dpath, setupParam['input']['truthFile'])

    plotdir = setupParam['output']['plotdir']
    MLplotdir = setupParam['output']['MLplotdir']

    logreg = setupParam['ML']['logreg']
    SVM = setupParam['ML']['SVM']
    RFC = setupParam['ML']['RFC']

    saveFig = setupParam['misc']['saveFig']
    verbose = setupParam['misc']['verbose']

    trainhue = setupParam['ML']['trainhue']
    trainndvi = setupParam['ML']['trainndvi']
    trainendvi = setupParam['ML']['trainendvi']
    traincvi = setupParam['ML']['traincvi']
    trainng = setupParam['ML']['trainng']
    trainnnir = setupParam['ML']['trainnnir']
    trainnr = setupParam['ML']['trainnr']
    traintvi = setupParam['ML']['traintvi']

