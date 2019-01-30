from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("-s", default='/home/meefly/data/results/multi_passage', type=str,
                    help="")
args = parser.parse_args()
import_submodules('src')
train_model_from_file("./net/vnet_dureader.jsonnet", args.s, force=True)
