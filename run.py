from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules

import_submodules('src')
train_model_from_file("./net/vnet.jsonnet", "/home/meefly/data/results/multi_passage", force=True)
