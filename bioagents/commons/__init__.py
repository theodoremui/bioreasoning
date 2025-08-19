"""
bioagents.commons
=================

Shared utilities and helpers used across the BioAgents codebase.

Currently exported:
- ``classproperty``: a descriptor that provides a read-only, lazily-evaluated
  property at the class level, similar to ``@property`` for instances.
"""

from typing import Any, Callable, Generic, Optional, Type, TypeVar

_T = TypeVar("_T")


class classproperty(Generic[_T]):  # noqa: N801 - intentionally camelCase for parity with @property
    """A class-level property descriptor.

    This behaves like ``@property`` but for classes instead of instances. It is
    accessed on the class and returns a value derived from the class itself
    (commonly used for cached, lazily initialized singletons held in class vars).

    Usage:
        >>> class C:
        ...     _value = 41
        ...
        ...     @classproperty
        ...     def value(cls):
        ...         # cls refers to the class (not an instance)
        ...         return cls._value + 1
        ...
        >>> C.value
        42

    Notes:
    - Only read access is supported (mirrors common read-only property usage).
    - The getter receives the class object (``cls``) rather than an instance.
    - This is intentionally minimal and does not implement setter/deleter.
    """

    __slots__ = ("_fget",)

    def __init__(self, fget: Callable[[Type[Any]], _T]):
        if not callable(fget):
            raise TypeError("classproperty requires a callable getter")
        self._fget = fget

    def __get__(self, instance: Optional[object], owner: Type[Any]) -> _T:  # type: ignore[override]
        # "owner" is the class object when accessed as C.attr or c.__class__.attr
        return self._fget(owner)


__all__ = ["classproperty"]


