from b737wing.tools.pin_pcores import ALL, cores_for


def test_fewer_than_20_cores_get_ranked_p_cores_first():
    assert cores_for(1) == [6]
    assert cores_for(4) == [6, 7, 8, 18]
    assert cores_for(8) == sorted(cores_for(8), key=[6, 7, 8, 18, 9, 19, 0, 1].index)
    assert set(cores_for(10)) >= {0, 1, 6, 7, 8, 9, 18, 19}


def test_twenty_or_more_uses_all_cores():
    assert cores_for(20) == ALL and cores_for(32) == ALL and len(ALL) == 20
