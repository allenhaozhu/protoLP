import os
import pickle
import numpy as np
import torch
import math
# from tqdm import tqdm

# ========================================================
#   Usefull paths
# _datasetFeaturesFiles = {"miniimagenet": "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk",
#                          "miniimagenet": "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk",
#                          "cub": "./checkpoints/CUB/WideResNet28_10_S2M2_R/last/output.plk",
#                          "cifar": "./checkpoints/cifar/WideResNet28_10_S2M2_R/last/output.plk",
#                          "cross": "./checkpoints/cross/WideResNet28_10_S2M2_R/last/output.plk"}
_datasetFeaturesFiles = {"miniimagenet": "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk",
                         "miniimagenet_other": "./checkpoints/mini_wideres_best_test.pickle",
                         "Res12_miniimagenet": "./checkpoints/miniImagenet/resnet12_S2M2_R/last/output.plk",
                         "Res12x3_miniimagenet": "./checkpoints/miniimagenet5.pkl",
                         "Res12AS_miniimagenet": "./checkpoints/miniAS.pkl",
                         "Res18_miniimagenet": "./checkpoints/mini_resnet18_best_test.pickle",
                         "Res18_mirror_miniimagenet": "./checkpoints/mini_mirror_resnet18_best_test.pickle",
                         "densenet_miniimagenet": "./checkpoints/mini_densenet_best_test.pickle",
                         "tierdimagenet": "./checkpoints/tieredImagenet/WideResNet28_10_S2M2_R/last/output.plk",
                         "tierdimagenet_other": "./checkpoints/tiered_wideres_best_test.pickle",
                         "Res12_tierdimagenet": "./checkpoints/tieredImagenet/resnet12_S2M2_R/last/output.plk",
                         "Res12x3_tierdimagenet": "./checkpoints/tiered5.pkl",
                         "Res12AS_tierdimagenet": "./checkpoints/tieredAS.pkl",
                         "Res18_tierdimagenet": "./checkpoints/tiered_resnet18_best_test.pickle",
                         "densenet_tierdimagenet": "./checkpoints/tiered_densenet_best_test.pickle",
                         "cub": "./checkpoints/CUB/WideResNet28_10_S2M2_R/last/output.plk",
                         "Res12_cub": "./checkpoints/CUB/resnet12_S2M2_R/last/output.plk",
                         "Res12x3_cub": "./checkpoints/CUB5.pkl",
                         "Res12AS_cub": "./checkpoints/cubAS.pkl",
                         "cifar": "./checkpoints/cifar/WideResNet28_10_S2M2_R/last/output.plk",
                         "Res12x3_cifar": "./checkpoints/cifar5.pkl",
                         "Res12AS_cifar": "./checkpoints/cifarAS.pkl",
                         "Res12_cifar": "./checkpoints/cifar/resnet12_S2M2_R/last/output.plk",
                         "cross": "./checkpoints/cross/WideResNet28_10_S2M2_R/last/output.plk"}
_cacheDir = "./cache"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None

def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)

# def _load_pickle(file):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         if data.__len__() == 2:
#             data = data[1]
#         labels = [np.full(shape=len(data[key]), fill_value=key)
#                   for key in data]
#         data = [features for key in data for features in data[key]]
#         dataset = dict()
#         dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
#         dataset['labels'] = torch.LongTensor(np.concatenate(labels))
#         return dataset

def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        if data.__len__() == 2:
            data = data[1]
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset

# def _load_pickle(file):
#     with open(file, 'rb') as f:
#         dataset = pickle.load(f)
#         # if data.__len__() == 2:
#         #     data = data[1]
#         # labels = [np.full(shape=len(data[key]), fill_value=key)
#         #           for key in data]
#         # data = [features for key in data for features in data[key]]
#         # dataset = dict()
#         # dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
#         # dataset['labels'] = torch.LongTensor(np.concatenate(labels))
#         return dataset
# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None

def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)

def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)

def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))

    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']+cfg['queries']]

    return dataset

def GenerateUnbalancedRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot'], data.shape[2]))
        label = torch.zeros(cfg['ways'],cfg['shot'],dtype=torch.int64)
    alpha = 2 * np.ones(cfg['ways'])
    pro = get_dirichlet_query_dist(alpha, 1, cfg['ways'], cfg['queries'] * cfg['ways'])[0]
    querySet = []
    labelSet = []
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)

        if generate:
            # dataset[i] = data[classes[i], shuffle_indices,
            #                   :][:cfg['shot']+cfg['queries']]
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']]
            if pro[i] > data[classes[i], shuffle_indices,:].shape[0]:
                dist = pro[i] - data[classes[i], shuffle_indices, :].shape[0]
                query = data[classes[i], shuffle_indices,:][:pro[i]]
                query_extra = data[classes[i], shuffle_indices[:dist],:][:pro[i]]
                query = torch.cat((query,query_extra),dim=0)
            else:
                query = data[classes[i], shuffle_indices,:][:pro[i]]
            querySet.append(query)
            label[i] = i
            label_que = i*torch.ones(pro[i])
            labelSet.append(label_que)
    querys = torch.cat(querySet,dim=0)
    labels = torch.cat(labelSet,dim=0)
    dataset = torch.cat((dataset.reshape(-1,data.shape[2]),querys),dim=0)
    labels = torch.cat((label.reshape(-1),labels),dim=0)
    return dataset, labels

def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes

def setUnbalancedRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}_unbalanced".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg

def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg

def GenerateUnbalancedRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways']*(cfg['shot']+cfg['queries']), data.shape[2]))
    labels = torch.zeros((end-start, cfg['ways']*(cfg['shot']+cfg['queries'])),dtype=torch.int64)
    #get_dirichlet_query_dist(2, n_tasks, n_ways, q_shots)
    for iRun in range(end-start):
        dataset[iRun], labels[iRun]= GenerateUnbalancedRun(start+iRun, cfg)

    return dataset, labels

def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    #get_dirichlet_query_dist(2, n_tasks, n_ways, q_shots)
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)

    return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
