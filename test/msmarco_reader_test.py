from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

# export PYTHONPATH=/home/meelfy/working/msmarco/:$PYTHONPATH
# export PYTHONPATH=/home/meelfy/working/msmarco/src/:$PYTHONPATH
from src.msmarco_reader import MsmarcoMultiPassageReader


class TestQAReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MsmarcoMultiPassageReader(lazy=True)
        instances = ensure_list(reader.read('../fixtures/2samples.json'))

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["question"].tokens] == 'average cost of assisted living in washington state'.split()
        assert len(fields['spans_start']) == 10
        assert len(fields['spans_end']) == 10
        assert len(fields['spans_start'][0]) == 1

        fields = instances[1].fields
        passage = ['According to the 2015 Genworth Financial cost of care survey, the average cost of assisted living in Washington is $4,625 per month. The monthly base rate for Washington assisted living is typically higher when compared to neighboring states.', 'According to the 2015 Genworth Financial cost of care survey, the average cost of assisted living in Washington is $4,625 per month. The monthly base rate for Washington assisted living is typically higher when compared to neighboring states. Washington is also more expensive compared to the national average.', 'Just as the cost of real estate varies by geographic area, assisted living costs also vary nationwide. In the 2015 Cost of Care Survey conducted by Genworth Financial, assisted living showed an increase of 2.86% compared to the 2014 costs and the national median monthly rate is $3,600.', 'The average annual cost of a private assisted living bedroom in Washington is 38,410 dollars. This makes it the 11th most expensive state for assisted living care in the country.', 'In the 2015 Cost of Care Survey conducted by Genworth Financial, assisted living showed an increase of 2.86% compared to the 2014 costs and the national median monthly rate is $3,600.', 'The 2400 assisted living facilities in Washington are regulated by the Aging and Disability Services. The average annual cost of a private assisted living bedroom in Washington is 38,410 dollars.', 'The average cost of Assisted Living in Washington is $4,625. Assisted Living costs range from $1,845 to $9,750 depending on location and other factors.', 'Assisted Living Costs by State. The Genworth 2015 Cost of Care Survey is the most comprehensive study of its kind. Genworth Financial surveyed approximatey 15% of assisted living communities. The monthly cost is for a one-bedroom unit in an assisted living facility.', 'Assisted Living Costs in Olympia, WA. Cost for a single bedroom in Olympia runs between $2,550 to $3,795 a month. Availability is almost guaranteed but you can request information online by filling out the form to the right or call us at (866) 333-8391 for a no-cost, in-depth assessment of your senior care needs.', 'In Washington there are 461 Assisted Living Facilities. We can help you find the best matches for your needs. The average cost of Assisted Living in Washington is $4,625 per month.']
        start = fields['spans_start'][6][0].sequence_index
        end = fields['spans_end'][6][0].sequence_index
        assert passage == fields['metadata']['passage_tokens'][6]
        assert len(fields['metadata']['answer_texts']) == 10
        assert ''.join(passage[start:end+1]) == fields['metadata']['answer_texts'][6][0].replace(' ','')
