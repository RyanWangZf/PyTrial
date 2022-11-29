import sdmetrics


def _select_metrics(metrics):
    '''
    Select multiple metrics at once.

    Parameters
    ----------

    metrics (list[str]):
        List of metric names to apply or None:
            metrics = [
                'BNLogLikelihood',
                'LogisticDetection',
                'SVCDetection',
                'BinaryDecisionTreeClassifier',
                'BinaryAdaBoostClassifier',
                'BinaryLogisticRegression',
                'BinaryMLPClassifier',
                'MulticlassDecisionTreeClassifier',
                'MulticlassMLPClassifier',
                'LinearRegression',
                'MLPRegressor',
                'GMLogLikelihood',
                'CSTest',
                'KSComplement',
                'StatisticSimilarity',
                'BoundaryAdherence',
                'MissingValueSimilarity',
                'CategoryCoverage',
                'TVComplement',
                'CategoricalCAP',
                'CategoricalZeroCAP',
                'CategoricalGeneralizedCAP',
                'CategoricalNB',
                'CategoricalKNN',
                'CategoricalRF',
                'CategoricalSVM',
                'CategoricalEnsemble',
                'NumericalLR',
                'NumericalMLP',
                'NumericalSVR',
                'NumericalRadiusNearestNeighbor',
                'ContinuousKLDivergence',
                'DiscreteKLDivergence',
                'ContingencySimilarity',
                'CorrelationSimilarity',
            ]

    Returns
    -------
    final_metrics: sdmetrics.single_table.SingleTableMetric
    '''
    metric_classes = sdmetrics.single_table.SingleTableMetric.get_subclasses()
    if metrics is None:
        metric_classes = {
            'KSComplement': metric_classes['KSComplement'],
            'CSTest': metric_classes['CSTest'],
        }
        return metric_classes

    final_metrics = {}
    for metric in metrics:
        if isinstance(metric, str):
            try:
                final_metrics[metric] = metric_classes[metric]
            except KeyError:
                raise ValueError(f'Unknown metric: {metric}')

    return final_metrics

def evaluate(synthetic_data, real_data, metrics=None, aggregate=True):
    '''
    Apply multiple metrics at once.

    Parameters
    ----------
    synthetic_data (dict[str, pandas.DataFrame] or pandas.DataFrame):
        Map of names and tables of synthesized data. When evaluating a single table,
        a single ``pandas.DataFrame`` can be passed alone.

    real_data (dict[str, pandas.DataFrame] or pandas.DataFrame):
        Map of names and tables of real data. When evaluating a single table,
        a single ``pandas.DataFrame`` can be passed alone.
        If metadata is None, this parameter must be a dataframe.

    metrics (list[str]):
        List of metric names to apply:
            'BNLogLikelihood'<class 'sdmetrics.single_table.bayesian_network.BNLogLikelihood'>,
            'LogisticDetection': <class 'sdmetrics.single_table.detection.sklearn.LogisticDetection'>, 
            'SVCDetection': <class 'sdmetrics.single_table.detection.sklearn.SVCDetection'>, 
            'BinaryDecisionTreeClassifier': <class 'sdmetrics.single_table.efficacy.binary.BinaryDecisionTreeClassifier'>, 
            'BinaryAdaBoostClassifier': <class 'sdmetrics.single_table.efficacy.binary.BinaryAdaBoostClassifier'>, 
            'BinaryLogisticRegression': <class 'sdmetrics.single_table.efficacy.binary.BinaryLogisticRegression'>, 
            'BinaryMLPClassifier': <class 'sdmetrics.single_table.efficacy.binary.BinaryMLPClassifier'>, 
            'MulticlassDecisionTreeClassifier': <class 'sdmetrics.single_table.efficacy.multiclass.MulticlassDecisionTreeClassifier'>, 
            'MulticlassMLPClassifier': <class 'sdmetrics.single_table.efficacy.multiclass.MulticlassMLPClassifier'>, 
            'LinearRegression': <class 'sdmetrics.single_table.efficacy.regression.LinearRegression'>, 
            'MLPRegressor': <class 'sdmetrics.single_table.efficacy.regression.MLPRegressor'>, 
            'GMLogLikelihood': <class 'sdmetrics.single_table.gaussian_mixture.GMLogLikelihood'>, 
            'CSTest': <class 'sdmetrics.single_table.multi_single_column.CSTest'>, 
            'KSComplement': <class 'sdmetrics.single_table.multi_single_column.KSComplement'>, 
            'StatisticSimilarity': <class 'sdmetrics.single_table.multi_single_column.StatisticSimilarity'>, 
            'BoundaryAdherence': <class 'sdmetrics.single_table.multi_single_column.BoundaryAdherence'>, 
            'MissingValueSimilarity': <class 'sdmetrics.single_table.multi_single_column.MissingValueSimilarity'>, 
            'CategoryCoverage': <class 'sdmetrics.single_table.multi_single_column.CategoryCoverage'>, 
            'TVComplement': <class 'sdmetrics.single_table.multi_single_column.TVComplement'>, 
            'CategoricalCAP': <class 'sdmetrics.single_table.privacy.cap.CategoricalCAP'>, 
            'CategoricalZeroCAP': <class 'sdmetrics.single_table.privacy.cap.CategoricalZeroCAP'>, 
            'CategoricalGeneralizedCAP': <class 'sdmetrics.single_table.privacy.cap.CategoricalGeneralizedCAP'>, 
            'CategoricalNB': <class 'sdmetrics.single_table.privacy.categorical_sklearn.CategoricalNB'>, 
            'CategoricalKNN': <class 'sdmetrics.single_table.privacy.categorical_sklearn.CategoricalKNN'>, 
            'CategoricalRF': <class 'sdmetrics.single_table.privacy.categorical_sklearn.CategoricalRF'>, 
            'CategoricalSVM': <class 'sdmetrics.single_table.privacy.categorical_sklearn.CategoricalSVM'>, 
            'CategoricalEnsemble': <class 'sdmetrics.single_table.privacy.ensemble.CategoricalEnsemble'>, 
            'NumericalLR': <class 'sdmetrics.single_table.privacy.numerical_sklearn.NumericalLR'>, 
            'NumericalMLP': <class 'sdmetrics.single_table.privacy.numerical_sklearn.NumericalMLP'>, 
            'NumericalSVR': <class 'sdmetrics.single_table.privacy.numerical_sklearn.NumericalSVR'>, 
            'NumericalRadiusNearestNeighbor': <class 'sdmetrics.single_table.privacy.radius_nearest_neighbor.NumericalRadiusNearestNeighbor'>, 
            'ContinuousKLDivergence': <class 'sdmetrics.single_table.multi_column_pairs.ContinuousKLDivergence'>, 
            'DiscreteKLDivergence': <class 'sdmetrics.single_table.multi_column_pairs.DiscreteKLDivergence'>, 
            'ContingencySimilarity': <class 'sdmetrics.single_table.multi_column_pairs.ContingencySimilarity'>, 
            'CorrelationSimilarity': <class 'sdmetrics.single_table.multi_column_pairs.CorrelationSimilarity'>,
    
    aggregate (bool):
        If ``get_report`` is ``False``, whether to compute the mean of all the normalized
        scores to return a single float value or return a ``dict`` containing the score
        that each metric obtained. Defaults to ``True``.
    
    Returns
    -------
    scores: float or sdmetrics.MetricsReport
    '''
    metrics = _select_metrics(metrics)
    scores = sdmetrics.compute_metrics(metrics, real_data, synthetic_data)

    if aggregate:
        return scores.normalized_score.mean()

    return scores
