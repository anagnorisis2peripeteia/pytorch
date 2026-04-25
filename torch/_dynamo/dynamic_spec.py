"""Dynamic shape specification types for ``torch.compile`` and ``torch.export``.

Provides class `IntSpec` for fine-grained control over whether an integer
(dimension size or scalar argument) is treated as static, backed, or unbacked
during compilation.

Backed vs. unbacked
-------------------
``torch.compile`` provides two kinds of dynamic shapes: ``backed`` and
``unbacked``. ``torch.compile`` guards on ``backed`` dynamic shapes and does
not provide a guarantee that no guards will be added to them. User code,
dynamo, inductor, and autograd all can add guards when tracing through
branching, e.g. ``if x.size() > 10``. Moreover, for 0/1 specializations,
backed symbols are specialized unconditionally to ``0``, ``1``, or ``>=2``
even without encountering a branching on those ranges.

On the contrary, ``unbacked`` dynamic shapes are guaranteed not to be guarded
on and are not 0/1 specialized. However, there is a possibility of throwing a
data-dependent error when a branch that requires their value is encountered
and no explicit unbacked handling is defined. The framework is converging to
a state where it won't throw DDE but rather pick general paths. One downside
of using unbacked is missed optimization opportunities due to either perf
bugs or picking general paths, or using a fixed non-example input-based hint.
An example of picking general paths is assuming input not contiguous in
functions called ``contiguous()`` and ``reshape()`` when it cannot be
symbolically proven, with a change of introducing a clone.

For more info see
https://dev-discuss.pytorch.org/t/backed-to-unbacked-from-guardable-to-guardless-shapes-in-pytorch/3333.
"""

import enum
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any, ClassVar


__all__ = ["IntSpecType", "IntSpec", "TensorSpec", "ModelSpec"]


class IntSpecType(enum.Enum):
    """How an integer should be treated during compilation.

    STATIC: compile-time constant; triggers recompilation if the value changes.
    BACKED: symbolic with guards and 0/1 specialization permitted.
    UNBACKED: symbolic, no guards, no 0/1 specialization; may raise a data-dependent error on branching.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Create via a classmethod factory or the constructor directly:

        IntSpec.static("x", value=10)
        IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        IntSpec("x", IntSpecType.STATIC, value=10)

    ``type`` is fixed at construction; all other fields are mutable via
    fluent setters that double as getters (no arg = read, one arg = write):

        spec = IntSpec.backed("batch", min=1, max=64)
        spec.guarding_hint(32)   # set, returns self
        spec.guarding_hint()     # get, returns 32
        spec.min(1).max(64)      # chain
    """

    _name: str | None
    _type: IntSpecType
    _min: int | None
    _max: int | None
    _value: int | None
    _guarding_hint: int | None
    _optimization_hint: int | None

    __slots__ = (
        "_name",
        "_type",
        "_min",
        "_max",
        "_value",
        "_guarding_hint",
        "_optimization_hint",
    )

    def __init__(
        self,
        name: str | None,
        type: IntSpecType,
        *,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        guarding_hint: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if not isinstance(type, IntSpecType):
            raise TypeError(f"IntSpec.type must be an IntSpecType, got {type!r}")
        self._type = type
        self._name = name
        self._min = min
        self._max = max
        self._value = value
        self._guarding_hint = guarding_hint
        self._optimization_hint = optimization_hint
        self._validate()

    def __setattr__(self, key: str, value: Any) -> None:
        # ``_type`` is the only pinned slot — it drives the per-mode
        # validation rules and integration-level dispatch (BACKED vs.
        # UNBACKED).
        if key == "_type" and hasattr(self, "_type"):
            raise AttributeError("IntSpec.type is immutable; cannot reassign")
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"IntSpec attribute {key!r} cannot be deleted")

    _MODE_KWARG_HINT: ClassVar[dict[IntSpecType, tuple[str, str]]] = {
        IntSpecType.STATIC: ("static", "value"),
        IntSpecType.BACKED: ("backed", "guarding_hint"),
        IntSpecType.UNBACKED: ("unbacked", "optimization_hint"),
    }

    @staticmethod
    def _check_name(value: Any, type_: IntSpecType) -> None:
        if value is not None and not isinstance(value, str):
            factory, kwarg = IntSpec._MODE_KWARG_HINT[type_]
            raise TypeError(
                f"IntSpec.name must be str or None, got "
                f"{value.__class__.__name__}; if you meant to pass a "
                f"value/hint, use a keyword argument "
                f"(e.g. IntSpec.{factory}({kwarg}={value!r}))"
            )

    @staticmethod
    def _check_int_field(field_name: str, value: Any) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise TypeError(
                f"IntSpec.{field_name} must be int or None, got "
                f"{value.__class__.__name__}"
            )

    # -- validation --------------------------------------------------------
    #
    # Single entry point: type checks (name, int fields), per-mode rules,
    # and cross-field invariants like ``min <= max``. Run on every
    # construction and on every fluent set (via ``_try_set``).

    def _validate(self) -> None:
        IntSpec._check_name(self._name, self._type)
        IntSpec._check_int_field("min", self._min)
        IntSpec._check_int_field("max", self._max)
        IntSpec._check_int_field("value", self._value)
        IntSpec._check_int_field("guarding_hint", self._guarding_hint)
        IntSpec._check_int_field("optimization_hint", self._optimization_hint)
        if self._type is IntSpecType.STATIC:
            if self._min is not None or self._max is not None:
                raise ValueError(
                    "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC"
                )
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        elif self._type is IntSpecType.BACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        else:  # UNBACKED
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
        if self._min is not None and self._max is not None and self._min > self._max:
            raise ValueError(
                f"min must be <= max, got min={self._min}, max={self._max}"
            )

    @classmethod
    def static(cls, name: str | None = None, *, value: int | None = None) -> "IntSpec":
        """Construct a STATIC `IntSpec`.

        ``value`` pins a concrete size; if ``None`` the value is taken from
        the example input at compile time.
        """
        return cls(name, type=IntSpecType.STATIC, value=value)

    @classmethod
    def backed(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        guarding_hint: int | None = None,
    ) -> "IntSpec":
        """Construct a BACKED `IntSpec`.

        ``guarding_hint`` is the concrete value the symbolic shape
        environment substitutes when a hint is needed for reasoning or
        codegen.
        """
        return cls(
            name,
            type=IntSpecType.BACKED,
            min=min,
            max=max,
            guarding_hint=guarding_hint,
        )

    @classmethod
    def unbacked(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> "IntSpec":
        """Construct an UNBACKED `IntSpec`.

        ``optimization_hint`` is used by downstream codegen (e.g. inductor
        autotuning) only; it never participates in symbolic reasoning.
        """
        return cls(
            name,
            type=IntSpecType.UNBACKED,
            min=min,
            max=max,
            optimization_hint=optimization_hint,
        )

    # -- identity (type is read-only) --------------------------------------

    @property
    def type(self) -> IntSpecType:
        return self._type

    # -- fluent get-or-set -------------------------------------------------
    #
    # Each method does double duty: no argument reads the current value,
    # one argument mutates in place + revalidates + returns ``self`` for
    # chaining. Per-mode validity is enforced on each set, so e.g.
    # ``IntSpec.static("x").guarding_hint(10)`` raises ``ValueError``.

    def _try_set(self, slot: str, new_value: Any) -> None:
        # Atomic: if ``_validate`` rejects the new state, roll back so the
        # spec stays in a consistent state for the caller.
        old = getattr(self, slot)
        setattr(self, slot, new_value)
        try:
            self._validate()
        except Exception:
            setattr(self, slot, old)
            raise

    def name(self, value: str | None = None) -> Any:
        if value is None:
            return self._name
        self._try_set("_name", value)
        return self

    def min(self, value: int | None = None) -> Any:
        if value is None:
            return self._min
        self._try_set("_min", value)
        return self

    def max(self, value: int | None = None) -> Any:
        if value is None:
            return self._max
        self._try_set("_max", value)
        return self

    def value(self, value: int | None = None) -> Any:
        if value is None:
            return self._value
        self._try_set("_value", value)
        return self

    def guarding_hint(self, value: int | None = None) -> Any:
        if value is None:
            return self._guarding_hint
        self._try_set("_guarding_hint", value)
        return self

    def optimization_hint(self, value: int | None = None) -> Any:
        if value is None:
            return self._optimization_hint
        self._try_set("_optimization_hint", value)
        return self

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._name is not None:
            parts.append(f"name={self._name!r}")
        parts.append(f"type={self._type.name}")
        if self._value is not None:
            parts.append(f"value={self._value}")
        if self._min is not None:
            parts.append(f"min={self._min}")
        if self._max is not None:
            parts.append(f"max={self._max}")
        if self._guarding_hint is not None:
            parts.append(f"guarding_hint={self._guarding_hint}")
        if self._optimization_hint is not None:
            parts.append(f"optimization_hint={self._optimization_hint}")
        return f"IntSpec({', '.join(parts)})"


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``IntSpec | None`` with length equal to the
    tensor's rank. ``None`` entries inherit the default dynamism policy from
    the compile context.

    Example::

        ts = TensorSpec(3)
        ts.set(0, IntSpec.backed("batch", min=1, max=64))
        # dims 1 and 2 are None -> inherit context default
    """

    def __init__(self, rank: int) -> None:
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        self._rank = rank
        self._specs: list[IntSpec | None] = [None] * rank

    @classmethod
    def from_list(cls, specs: list[IntSpec | None]) -> "TensorSpec":
        """Construct from an existing list of specs."""
        ts = cls(len(specs))
        ts._specs = list(specs)
        return ts

    @property
    def rank(self) -> int:
        return self._rank

    def set(self, index: int, spec: IntSpec) -> "TensorSpec":
        """Set the spec at ``index`` and return ``self`` for chaining."""
        self._specs[index] = spec
        return self

    def __getitem__(self, index: int) -> IntSpec | None:
        return self._specs[index]

    def __setitem__(self, index: int, spec: IntSpec | None) -> None:
        self._specs[index] = spec

    def __len__(self) -> int:
        return self._rank

    def __iter__(self) -> Iterator[IntSpec | None]:
        return iter(self._specs)

    def __repr__(self) -> str:
        specified = [
            f"{i}: {spec!r}" for i, spec in enumerate(self._specs) if spec is not None
        ]
        return f"TensorSpec(rank={self._rank}, {{{', '.join(specified)}}})"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec`'s design — specs
    # are immutable compile-time inputs compared via ``repr()`` when needed.


class ModelSpec:
    """Top-level dynamic-shape specification for a whole compiled model.

    A dict-like container mapping argument names (as they appear in the
    compiled function's signature) to per-argument specs. Per-argument spec
    can be:

    - :class:`TensorSpec` — per-dimension spec for a tensor argument.
    - :class:`IntSpec` — spec for a scalar integer argument.
    - ``dict[int, IntSpec | None]`` — sparse per-dim spec.
    - ``list[IntSpec | None]`` / ``tuple[IntSpec | None, ...]`` — positional
      per-dim spec.
    - ``None`` — inherit the compile-context default for that argument.

    Example::

        ModelSpec(
            {
                "x": TensorSpec(2).set(0, IntSpec.backed("batch")),
                "batch_size": IntSpec.backed("batch"),
            }
        )
    """

    def __init__(self, specs: dict[str, Any] | None = None) -> None:
        self._specs: dict[str, Any] = dict(specs) if specs else {}

    def set(self, name: str, spec: Any) -> "ModelSpec":
        """Assign *spec* to the argument *name*. Returns ``self`` for chaining."""
        self._specs[name] = spec
        return self

    def __getitem__(self, name: str) -> Any:
        return self._specs[name]

    def __setitem__(self, name: str, spec: Any) -> None:
        self._specs[name] = spec

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def items(self) -> Any:
        return self._specs.items()

    def get(self, name: str, default: Any = None) -> Any:
        return self._specs.get(name, default)

    def __repr__(self) -> str:
        return f"ModelSpec({self._specs!r})"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec` / :class:`TensorSpec`.


# ContextVar carrying the dynamic_shapes spec for the currently-running
# ``torch.compile``'d function. Set by :func:`_apply_dynamic_shapes` on each
# call; read by :func:`get_active_spec_for_dim` from inside the Dynamo
# variable builder during input wrapping. No tensor monkey-patching; no
# pre-installed guards — the spec directly selects ``DimDynamic`` in
# ``_automatic_dynamic``.
_active_dynamic_shapes: ContextVar[dict[str, Any] | None] = ContextVar(
    "_dynamo_active_dynamic_shapes", default=None
)


def _resolve_dim_spec(arg_spec: Any, dim: int) -> "IntSpec | None":
    """Extract the :class:`IntSpec` for *dim* from a per-argument spec.

    Supports the four forms accepted in ``dynamic_shapes``: ``TensorSpec``,
    ``dict[int, IntSpec]``, ``list``/``tuple`` of IntSpec-or-None, or an
    :class:`IntSpec` directly (for scalar-int arguments; ``dim`` ignored).
    """
    if isinstance(arg_spec, IntSpec):
        return arg_spec
    if isinstance(arg_spec, TensorSpec):
        return arg_spec[dim] if 0 <= dim < len(arg_spec) else None
    if isinstance(arg_spec, dict):
        return arg_spec.get(dim)
    if isinstance(arg_spec, (list, tuple)):
        return arg_spec[dim] if 0 <= dim < len(arg_spec) else None
    return None


def get_active_spec_for_arg(arg_name: str) -> Any:
    """Return the spec associated with *arg_name* in the active
    ``dynamic_shapes``, or ``None`` if no spec is active or the arg is not
    listed."""
    spec_dict = _active_dynamic_shapes.get()
    if spec_dict is None:
        return None
    if isinstance(spec_dict, ModelSpec):
        return spec_dict.get(arg_name)
    return spec_dict.get(arg_name)


def get_active_spec_for_dim(arg_name: str, dim: int) -> "IntSpec | None":
    """Return the :class:`IntSpec` for *dim* of argument *arg_name* in the
    active ``dynamic_shapes``, or ``None``."""
    arg_spec = get_active_spec_for_arg(arg_name)
    if arg_spec is None:
        return None
    return _resolve_dim_spec(arg_spec, dim)
