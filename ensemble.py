import pandas as pd
from utils import metrics_, AttrDict, read_yaml, parseargs


def ensemble_four_regions(vent, bsc, gm, wm, draw_mode='max'):
    vent = vent.rename(columns={'preds': 'preds_vent', 'probs': 'probs_vent'})
    gm = gm.rename(columns={'preds': 'preds_gm', 'probs': 'probs_gm'})
    bsc = bsc.rename(columns={'preds': 'preds_bsc', 'probs': 'probs_bsc'})
    wm = wm.rename(columns={'preds': 'preds_wm', 'probs': 'probs_wm'})

    vent = vent.sort_values(by='test_ses').reset_index(drop=True)
    bsc = bsc.sort_values(by='test_ses').reset_index(drop=True)
    wm = wm.sort_values(by='test_ses').reset_index(drop=True)
    gm = gm.sort_values(by='test_ses').reset_index(drop=True)
    four_reg = pd.concat(
        [gm, bsc[['preds_bsc', 'probs_bsc']], wm[['preds_wm', 'probs_wm']], vent[['preds_vent', 'probs_vent']]], axis=1)

    # majority voting
    only_pred = four_reg[['preds_bsc', 'preds_vent', 'preds_wm', 'preds_gm']]
    maj_vot = only_pred.mode(axis=1).values.tolist()

    only_probs = four_reg[['probs_bsc', 'probs_vent', 'probs_wm', 'probs_gm']]
    max_prob = only_probs.max(axis=1).values.tolist()
    model_max_prob = only_probs.idxmax(axis=1)
    max_ = ['preds_%s' % model.split('_')[1] for model in model_max_prob.values.tolist()]
    max_vot = [four_reg.iloc[i][max_[i]] for i in range(len(four_reg))]
    four_reg['max_prob'] = max_prob
    four_reg['max_vot'] = max_vot

    one_maj_vot = []
    for i, vot in enumerate(maj_vot):
        # taking max vot
        if vot[1] == 1:
            if draw_mode == 'max':
                one_maj_vot.append(max_vot[i])
            # taking max summation probabilities by predicted class
            elif draw_mode == 'sum':
                ex_pred = four_reg[four_reg.index == i][only_pred.columns]
                ex_prob = four_reg[four_reg.index == i][only_probs.columns]
                ex_pred.columns = range(only_pred.columns.size)
                ex_prob.columns = range(only_probs.columns.size)
                sum_0 = ex_prob[ex_pred == 0].sum(axis=1).values[0]
                sum_1 = ex_prob[ex_pred == 1].sum(axis=1).values[0]
                if sum_0 > sum_1:
                    one_maj_vot.append(vot[0])
                else:
                    one_maj_vot.append(vot[1])

        else:
            one_maj_vot.append(vot[0])

    four_reg['maj_vot'] = one_maj_vot
    metrics_(four_reg.maj_vot.values.tolist(), four_reg['class'].values.tolist())

    return four_reg


def ensemble_five_regions(vent, bsc, gm, wm, subc):
    vent = vent.rename(columns={'preds': 'preds_vent', 'probs': 'probs_vent'})
    gm = gm.rename(columns={'preds': 'preds_gm', 'probs': 'probs_gm'})
    bsc = bsc.rename(columns={'preds': 'preds_bsc', 'probs': 'probs_bsc'})
    wm = wm.rename(columns={'preds': 'preds_wm', 'probs': 'probs_wm'})
    subc = subc.rename(columns={'preds': 'preds_subc', 'probs': 'probs_subc'})

    vent = vent.sort_values(by='test_ses').reset_index(drop=True)
    bsc = bsc.sort_values(by='test_ses').reset_index(drop=True)
    wm = wm.sort_values(by='test_ses').reset_index(drop=True)
    gm = gm.sort_values(by='test_ses').reset_index(drop=True)
    subc = subc.sort_values(by='test_ses').reset_index(drop=True)
    five_reg = pd.concat([gm, bsc[['preds_bsc', 'probs_bsc']], wm[['preds_wm', 'probs_wm']],
                          vent[['preds_vent', 'probs_vent']], subc[['preds_subc', 'probs_subc']]], axis=1)

    # majority voting
    only_pred = five_reg[['preds_bsc', 'preds_vent', 'preds_wm', 'preds_gm', 'preds_subc']]
    maj_vot = only_pred.mode(axis=1).values.tolist()
    five_reg['maj_vot'] = [mv[0] for mv in maj_vot]
    only_probs = five_reg[['probs_bsc', 'probs_vent', 'probs_wm', 'probs_gm', 'probs_subc']]
    max_prob = only_probs.max(axis=1).values.tolist()
    model_max_prob = only_probs.idxmax(axis=1)
    max_ = ['preds_%s' % model.split('_')[1] for model in model_max_prob.values.tolist()]
    max_vot = [five_reg.iloc[i][max_[i]] for i in range(len(five_reg))]
    five_reg['max_prob'] = max_prob
    five_reg['max_vot'] = max_vot

    metrics_(five_reg.maj_vot.values.tolist(), five_reg['class'].values.tolist())

    return five_reg, max_


if __name__ == '__main__':
    # set general options
    pars = parseargs()
    cfg = read_yaml(pars.cfg_file)
    opts = AttrDict(cfg)

    # load prediction files
    file_vent = pd.read_csv(pars.pred_vent)
    file_bsc = pd.read_csv(pars.pred_bsc)
    file_wm = pd.read_csv(pars.pred_wm)
    file_gm = pd.read_csv(pars.pred_gm)
    file_sgm = pd.read_csv(pars.pred_sgm)

    df_ensb = ensemble_five_regions(file_vent, file_bsc, file_gm, file_wm, file_sgm)
