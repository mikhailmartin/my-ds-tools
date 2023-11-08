from ._eda_distribution import (
    cat_feature_report, num_feature_report, na_bar_plot, target_distribution_plot)
from ._eda_datashift import area_plot, ridge_plot


def unsupported_calplot(*args, **kwargs):
    raise ImportError('Не установлен calplot. Нажми `pip install calplot`))')


try:
    import calplot
    have_calplot = True
except ImportError:
    have_calplot = False

if have_calplot:
    from ._eda_datashift import na_datashift
else:
    na_datashift = unsupported_calplot

__all__ = [
    # distribution
    'cat_feature_report',
    'num_feature_report',
    'na_bar_plot',
    'target_distribution_plot',
    # data_shift
    'area_plot',
    'na_datashift',
    'ridge_plot',
]
