# encoding: utf-8

# export PYTHONPATH=/home/meelfy/working/msmarco/:$PYTHONPATH
from allennlp.common.testing import ModelTestCase
from src.msmarco_reader import MsmarcoMultiPassageReader
from src.vnet import VNet

class VnetTest(ModelTestCase):
    def setUp(self):
        super(VnetTest, self).setUp()
        self.set_up_model('./vnet.jsonnet',
                          '../fixtures/2samples.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)