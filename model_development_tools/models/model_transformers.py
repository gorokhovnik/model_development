from typing import List

import numpy as np

import lightgbm as lgb

import re


def join_lgb_boosters(
        boosters: List[lgb.Booster],
) -> lgb.Booster:
    n_boosters = len(boosters)
    params = boosters[0].params

    boosters_txt = []
    for booster in boosters:
        boosters_txt += [booster.model_to_string()]

    feature_infos, tree_sizes, trees, feature_importances = [], [], '', {}

    for booster in boosters_txt:
        cur = booster.split('\n')

        for idx, c in enumerate(cur):
            if c[:5] == 'Tree=':
                meta = cur[:idx]
                trees += re.sub('Tree=[0-9]+',
                                lambda a: 'Tree=' + str(int(a.group()[5:]) + len(tree_sizes)),
                                '\n'.join(cur[idx:cur.index('end of trees')]) + '\n')
                break

        parameters = '\n'.join(cur[cur.index('parameters:'):])

        feat_imp = {i[:i.index('=')]: int(i[i.index('=') + 1:])
                    for i in cur[cur.index('feature_importances:') + 1:cur.index('parameters:') - 1]}
        for i in feat_imp:
            if i in feature_importances:
                feature_importances[i] += feat_imp[i]
            else:
                feature_importances[i] = feat_imp[i]

        for c in cur:
            if c[:14] == 'feature_infos=':
                feature_infos += [[[float(j) if j != 'on' else j
                                    for j in i[1:-1].split(':')] for i in c[14:].split(' ')]]
                break

        for c in cur:
            if c[:11] == 'tree_sizes=':
                tree_sizes += c[11:].split(' ')
                break

    feature_infos = [[np.min([feature_infos[j][i][0] if feature_infos[j][i][0] != 'on' else np.inf
                              for j in range(len(feature_infos))]),
                      np.max([feature_infos[j][i][1] if feature_infos[j][i][0] != 'on' else -np.inf
                              for j in range(len(feature_infos))])]
                     for i in range(len(feature_infos[0]))]

    feature_infos = 'feature_infos=' + ' '.join(['[' + re.sub('.0:', ':', str(i[0]) + ':')
                                                 + re.sub('.0]', ']', str(i[1]) + ']')
                                                 if i != [np.inf, -np.inf] else 'none' for i in feature_infos])

    trees = trees.split('\n')
    for idx, c in enumerate(trees):
        if c[:11] == 'leaf_value=':
            trees[idx] = 'leaf_value=' + ' '.join([str(float(cc) / n_boosters) for cc in c[11:].split(' ')])
    trees = '\n'.join(trees)

    tree_sizes = 'tree_sizes=' + ' '.join([str(len(i) + 3) for i in trees.split('\n\n\n')[:-1]])

    for idx, m in enumerate(meta):
        if m[:14] == 'feature_infos=':
            meta[idx] = feature_infos
        if m[:11] == 'tree_sizes=':
            meta[idx] = tree_sizes
    meta = '\n'.join(meta)
    trees += 'end of trees\n'

    feature_importances = 'feature_importances:\n' + '\n'.join([i + '=' + str(feature_importances[i])
                                                                for i in sorted(feature_importances,
                                                                                key=feature_importances.get,
                                                                                reverse=True)]) + '\n'

    superbooster_str = '\n'.join([meta, trees, feature_importances, parameters])
    superbooster = lgb.Booster(model_str=superbooster_str, silent=True)
    superbooster_str = superbooster.model_to_string()
    superbooster = lgb.Booster(model_str=superbooster_str, silent=True)
    superbooster.params = params

    return superbooster
