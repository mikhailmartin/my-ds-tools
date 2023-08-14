from ..eda import ridge_plot, area_plot
from .forget_weights import get_forget_weights


def unsupported_calplot(*args, **kwargs):
    raise ImportError('Не установлен calplot. Нажми `pip install calplot`))')


try:
    import calplot
    have_calplot = True
except ImportError:
    have_calplot = False

if have_calplot:
    from ..eda import na_datashift
else:
    na_datashift = unsupported_calplot


__all__ = [
    # eda
    'area_plot',
    'ridge_plot',
    'na_datashift',

    'get_forget_weights',
]
