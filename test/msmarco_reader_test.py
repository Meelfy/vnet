from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

# export PYTHONPATH=/home/meelfy/working/msmarco/:$PYTHONPATH
# export PYTHONPATH=/home/meelfy/working/msmarco/src/:$PYTHONPATH
from src.msmarco_reader import MsmarcoMultiPassageReader

class TestQAReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MsmarcoMultiPassageReader(lazy=True)
        instances = ensure_list(reader.read('../fixtures/2samples.json'))

        assert len(instances) == 2
        fields = instances[0].fields
        assert [t.text for t in fields["question"].tokens] == [")","what","was","the","immediate","impact","of","the","success","of","the","manhattan","project","?"]
        assert len(fields['spans_start']) == 10
        assert len(fields['spans_end']) == 10
        assert len(fields['spans_start'][0]) == 1

        fields = instances[1].fields
        passage = "The approach is based on a theory of justice that considers crime and wrongdoing to be an offense against an individual or community , rather than the State . Restorative justice that fosters dialogue between victim and offender has shown the highest rates of victim satisfaction and offender accountability .".split(' ')
        start = fields['spans_start'][6][0].sequence_index
        end = fields['spans_end'][6][0].sequence_index

        assert passage == fields['metadata']['passage_tokens'][6]
        assert len(fields['metadata']['answer_texts']) == 10
        assert ''.join(passage[start:end+1]) == fields['metadata']['answer_texts'][6][0].replace(' ','')