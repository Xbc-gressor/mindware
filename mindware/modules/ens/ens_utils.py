
def better_ens(ens1, ens2):

    assert ens1 is not None or ens2 is not None

    if ens1 is None or ens2 is None:

        return ens2 is None

    if 'val' in ens1:
        if ens1['val'] != ens2['val']:
            return ens1['val'] > ens2['val']

    if 'train' in ens1:
        if ens1['train'] != ens2['train']:
            return ens1['train'] > ens2['train']

    if 'ensemble_size' in ens1 and 'ratio' in ens1:
        if ens1['ensemble_size'] * ens1['ratio'] != ens2['ensemble_size'] * ens2['ratio']:
            return ens1['ensemble_size'] * (ens1['ratio'] + 1) > ens2['ensemble_size'] * (ens2['ratio'] + 1)

    if 'stack_layers' in ens1:
        if ens1['stack_layers'] != ens2['stack_layers']:
            return ens1['stack_layers'] < ens2['stack_layers']

    return False