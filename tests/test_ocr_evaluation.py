import pytest
from sensegraph.ocr_evaluation import region_text_counts


def test_missed_extra_and_wrong_words_stay_in_denominators():
    result=region_text_counts(['12','X','junk'],['12','Y','missing'],[0,1,-1])
    assert result=={'gt_words':3,'pred_words':3,'correct_words':1,'matched':2,
                    'gt_characters':10,'character_errors':12,'matched_characters':3,'matched_errors':1}
    assert region_text_counts([],['0'],[])['character_errors']==1
    assert region_text_counts(['extra'],[],[-1])['character_errors']==5


def test_invalid_matching_is_not_silently_scored():
    with pytest.raises(ValueError): region_text_counts(['a','a'],['a'],[0,0])
    with pytest.raises(ValueError): region_text_counts(['a'],['a'],[1])
