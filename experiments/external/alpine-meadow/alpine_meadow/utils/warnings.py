# pylint: disable=consider-using-f-string

"""Test Utility functions. Copy from sklearn."""
import sys
from functools import wraps
import warnings


def clean_warning_registry():
    """Clean Python warning registry for easier testing of warning messages.
    We may not need to do this any more when getting rid of Python 2, not
    entirely sure. See https://bugs.python.org/issue4180 and
    https://bugs.python.org/issue21724 for more details.
    """
    reg = "__warningregistry__"
    for _, mod in list(sys.modules.items()):
        if hasattr(mod, reg):
            try:
                getattr(mod, reg).clear()
            except AttributeError:
                pass


def ignore_warnings(obj=None, category=Warning):
    """Context manager and decorator to ignore warnings.
    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.
    Parameters
    ----------
    obj : callable or None
        callable where you want to ignore the warnings.
    category : warning class, defaults to Warning.
        The category to filter. If Warning, all categories will be muted.
    Examples
    --------
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')
    >>> def nasty_warn():
    ...    warnings.warn('buhuhuhu')
    ...    print(42)
    >>> ignore_warnings(nasty_warn)()
    42
    """
    if isinstance(obj, type) and issubclass(obj, Warning):  # pylint: disable=no-else-raise
        # Avoid common pitfall of passing category as the first positional
        # argument which result in the test not being run
        warning_name = obj.__name__
        raise ValueError(
            "'obj' should be a callable where you want to ignore warnings. "
            "You passed a warning class instead: 'obj={warning_name}'. "
            "If you want to pass a warning class to ignore_warnings, "
            "you should use 'category={warning_name}'".format(
                warning_name=warning_name))
    elif callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        return _IgnoreWarnings(category=category)


class _IgnoreWarnings:
    """Improved and simplified Python warnings context manager and decorator.
    This class allows the user to ignore the warnings raised by a function.
    Copied from Python 2.7.5 and modified as required.
    Parameters
    ----------
    category : tuple of warning class, default to Warning
        The category to filter. By default, all the categories will be muted.
    """

    def __init__(self, category):
        self._record = True
        self._module = sys.modules['warnings']
        self._entered = False
        self.log = []
        self.category = category

    def __call__(self, fn):
        """Decorator to catch and hide warnings without visual nesting."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            clean_warning_registry()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", self.category)
                return fn(*args, **kwargs)

        return wrapper

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules['warnings']:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        clean_warning_registry()
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning
        self.log[:] = []
        clean_warning_registry()
