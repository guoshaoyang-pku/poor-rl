import pytest

from rlforge.curriculum import parse, resolve


def test_update_stages():
    assert resolve('8:60,16:60,32:0', 0)['G'] == 8
    assert resolve('8:60,16:60,32:0', 0, 40)['max_steps'] == 20
    r = resolve('8:60,16:60,32:0', 2, open_ended_steps=500)
    assert (r['G'], r['max_steps'], r['last']) == (32, 500, True)


def test_epoch_stages_convert_to_updates():
    r = resolve('4@1,8@1,16@1,32@0', 1, rows=25071, batch=2048)
    assert (r['G'], r['max_steps']) == (8, 98)  # ceil(25071 * 8 / 2048)


@pytest.mark.parametrize('spec', ['', '8:0,16:60', '0:10', '8:-1'])
def test_bad_specs(spec):
    with pytest.raises(ValueError):
        parse(spec)


def test_errors():
    with pytest.raises(ValueError):
        resolve('8:60', 0, 60)  # stage already complete
    with pytest.raises(ValueError):
        resolve('24:10', 0, batch=2048)  # B not a multiple of G
    with pytest.raises(ValueError):
        resolve('8@1', 0)  # epoch unit needs rows and batch
