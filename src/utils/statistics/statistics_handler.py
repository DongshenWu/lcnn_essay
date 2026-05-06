import pandas
import matplotlib.pyplot as plt
from itertools import filterfalse
from math import ceil
from warnings import warn


class Statistics:
    vars_aggs: dict
    vars_noaggs: dict
    data: pandas.DataFrame

    def __init__(self, vars_aggs: dict[str: str]) -> None:
        """Per-epoch metric accumulator.

        `vars_aggs` maps column name to pandas aggregation: a string ('mean',
        'median'), a 'lambda x: ...' string (eval'd), or 'none' for scalars
        passed through unaggregated (e.g. epoch number).
        """
        def not_requires_agg(args): return args[1] == 'none'
        def requires_lambda_agg(args): return 'lambda' in args[1]
        def requires_str_agg(args): return not not_requires_agg(
            args) and not requires_lambda_agg(args)

        vars_aggs_str = dict(filter(requires_str_agg, vars_aggs.items()))
        vars_aggs_lambda = dict(filter(requires_lambda_agg, vars_aggs.items()))
        vars_aggs_lambda = {k: eval(v) for k, v in vars_aggs_lambda.items()}
        self.vars_aggs = {**vars_aggs_str, **vars_aggs_lambda}
        self.vars_noaggs = dict(filter(not_requires_agg, vars_aggs.items()))
        self.data = pandas.DataFrame(columns=vars_aggs.keys())

    def requires_agg(self, args):
        k, v = args
        return k in self.vars_aggs.keys()

    def to_series(self, args):
        k, v = args
        return k, pandas.Series(v)

    def update(self, **kwargs) -> None:
        values_agg = dict(map(self.to_series, filter(self.requires_agg, kwargs.items())))
        values_noagg = dict(filterfalse(self.requires_agg, kwargs.items()))

        def isin_kwargs(args): return args[0] in kwargs.keys()
        agg_maps = dict(filter(isin_kwargs, self.vars_aggs.items()))

        values_dict = pandas.DataFrame(values_agg).agg(agg_maps).to_dict()
        values_dict = {**values_noagg, **values_dict}
        df_actual = pandas.DataFrame(values_dict, index=[pandas.Timestamp.now()])
        self.data = pandas.concat([self.data, df_actual])

    def get_last(self):
        return self.data.iloc[-1, :].to_dict()

    def save(self, path: str):
        self.data.to_csv(path)

    def save_plot(self, path: str, subplots=True):
        if pandas.__version__ != '1.5.3':
            return None

        cols = self.data.columns
        train_metrics = [name for name in cols if 'train' in name]
        val_metrics = [name for name in cols if 'val' in name]
        train_metrics.sort()
        val_metrics.sort()
        plot_y = list(zip(train_metrics, val_metrics))
        if not subplots:
            warn('Warning: subplots=False has not been implemented yet.')
        # fit the subplots into a square layout
        n_plots = 1+len(plot_y)
        n_rows = ceil(n_plots**0.5)
        n_cols = ceil(n_plots / n_rows)

        ax = self.data.plot(subplots=plot_y, layout=(n_rows, n_cols))

        plt.savefig(path)

    def __repr__(self) -> str:
        vars_aggs = {**self.vars_aggs, **self.vars_noaggs}
        repr = 'Metric Aggreggation Map:\n'
        repr += str(vars_aggs).replace(',', '\n') + '\n'
        return repr + self.data.__repr__()
