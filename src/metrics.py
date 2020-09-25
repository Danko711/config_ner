from itertools import chain
from logging import getLogger

import numpy as np

log = getLogger(__name__)

def ner_token_f1(y_true, y_pred, print_results=False):


    y_true = list(chain(*y_true))
    y_pred = list(chain(*y_pred))



    # Drop BIO or BIOES markup
    assert all(len(str(tag).split('-')) <= 2 for tag in y_true)

    y_true = [str(tag).split('-')[-1] for tag in y_true]
    y_pred = [str(tag).split('-')[-1] for tag in y_pred]
    tags = set(y_true) | set(y_pred)
    tags_dict = {tag: n for n, tag in enumerate(tags)}

    y_true_inds = np.array([tags_dict[tag] for tag in y_true])
    y_pred_inds = np.array([tags_dict[tag] for tag in y_pred])

    results = {}
    for tag, tag_ind in tags_dict.items():
        if tag == 'O':
            continue
        tp = np.sum((y_true_inds == tag_ind) & (y_pred_inds == tag_ind))
        fn = np.sum((y_true_inds == tag_ind) & (y_pred_inds != tag_ind))
        fp = np.sum((y_true_inds != tag_ind) & (y_pred_inds == tag_ind))
        n_pred = np.sum(y_pred_inds == tag_ind)
        n_true = np.sum(y_true_inds == tag_ind)
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag] = {'precision': precision, 'recall': recall,
                        'f1': f1, 'n_true': n_true, 'n_pred': n_pred,
                        'tp': tp, 'fp': fp, 'fn': fn}

    results['__total__'], accuracy, total_true_entities, total_predicted_entities, total_correct = _global_stats_f1(
        results)
    n_tokens = len(y_true)
    if print_results:
        log.debug('TOKEN LEVEL F1')
        _print_conll_report(results, accuracy, total_true_entities, total_predicted_entities, n_tokens, total_correct)
    return results['__total__']['f1']