import pandas
import numpy
from rep.metaml import FoldingClassifier
from rep.data import LabeledDataStorage


def train_one_vs_one(base_estimators, data_b, data_c, data_light,
                     prefix='bdt', n_folds=2, folding=True, features=None, profile=None):

    data_b_c_lds = LabeledDataStorage(pandas.concat([data_b, data_c]), [1] * len(data_b) + [0] * len(data_c))
    data_c_light_lds = LabeledDataStorage(pandas.concat([data_c, data_light]), [1] * len(data_c) + [0] * len(data_light))
    data_b_light_lds = LabeledDataStorage(pandas.concat([data_b, data_light]), [1] * len(data_b) + [0] * len(data_light))

    if folding:        
        tt_folding_b_c = FoldingClassifier(base_estimators[0], n_folds=n_folds, random_state=11, parallel_profile=profile, 
                                           features=features)
        tt_folding_c_light = FoldingClassifier(base_estimators[1], n_folds=n_folds, random_state=11, parallel_profile=profile, 
                                               features=features)
        tt_folding_b_light = FoldingClassifier(base_estimators[2], n_folds=n_folds, random_state=11, parallel_profile=profile, 
                                               features=features)
    else:
        tt_folding_b_c = base_estimators[0]
        tt_folding_b_c.features = features
        tt_folding_c_light = base_estimators[1]
        tt_folding_c_light.features = features
        tt_folding_b_light = base_estimators[2]
        tt_folding_b_light.features = features
        
    tt_folding_b_c.fit_lds(data_b_c_lds)
    
    tt_folding_c_light.fit_lds(data_c_light_lds)

    tt_folding_b_light.fit_lds(data_b_light_lds)

    probs_b_c = numpy.concatenate([tt_folding_b_c.predict_proba(pandas.concat([data_b, data_c])),
                                   tt_folding_b_c.predict_proba(data_light)])[:, 1]
    probs_c_light = numpy.concatenate([tt_folding_c_light.predict_proba(data_b), 
                                       tt_folding_c_light.predict_proba(pandas.concat([data_c, data_light]))])[:, 1]
    probs_b_light = tt_folding_b_light.predict_proba(pandas.concat([data_b, data_light]))[:, 1]
    probs_b_light = numpy.concatenate([probs_b_light[:len(data_b)], tt_folding_b_light.predict_proba(data_c)[:, 1], 
                                       probs_b_light[len(data_b):]])
    
    additional_columns = pandas.DataFrame({prefix + '_b_c': probs_b_c,
                                           prefix + '_b_light': probs_b_light,
                                           prefix + '_c_light': probs_c_light})
    return additional_columns