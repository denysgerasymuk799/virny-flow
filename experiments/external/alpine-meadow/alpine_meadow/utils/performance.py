# pylint: disable=unused-import, invalid-name
"""Alpine Meadow performance utilities."""
import functools

from pyformance.registry import MetricsRegistry, get_qualname


registry = MetricsRegistry()


def timer(key):
    return registry.timer(key)


def time_calls(fn):
    """
    Decorator to time the execution of the function.

    :param fn: the function to be decorated
    :type fn: C{func}

    :return: the decorated function
    :rtype: C{func}
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _timer = registry.timer(f"call.{get_qualname(fn)}")
        with _timer.time(fn=get_qualname(fn)):
            return fn(*args, **kwargs)
    return wrapper


def histogram(key):
    return registry.histogram(key)


def hist_calls(fn):
    """
    Decorator to check the distribution of return values of a function.

    :param fn: the function to be decorated
    :type fn: C{func}

    :return: the decorated function
    :rtype: C{func}
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _histogram = registry.histogram(f"call.{get_qualname(fn)}")
        rtn = fn(*args, **kwargs)
        if isinstance(rtn, (int, float)):
            _histogram.update(rtn)
        return rtn
    return wrapper


def counter(key):
    return registry.counter(key)


def count_calls(fn):
    """
    Decorator to track the number of times a function is called.

    :param fn: the function to be decorated
    :type fn: C{func}

    :return: the decorated function
    :rtype: C{func}
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        registry.counter(f"call.{get_qualname(fn)}").inc()
        return fn(*args, **kwargs)
    return wrapper


def meter(key):
    return registry.meter(key)


def meter_calls(fn):
    """
    Decorator to the rate at which a function is called.

    :param fn: the function to be decorated
    :type fn: C{func}

    :return: the decorated function
    :rtype: C{func}
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        registry.meter(f"call.{get_qualname(fn)}").mark()
        return fn(*args, **kwargs)
    return wrapper


def dump_metrics():
    """
    Dump all the metrics as a dict.
    :return:
    """

    return registry.dump_metrics()
