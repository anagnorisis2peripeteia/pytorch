---
name: pyrefly-type-coverage
description: Migrate a file to use stricter Pyrefly type checking with annotations required for all functions, classes, and attributes.
---

# Pyrefly Type Coverage Skill

This skill guides you through improving type coverage in Python files using Pyrefly, Meta's type checker. Follow this systematic process to add proper type annotations to files.

## Prerequisites
- The file you're working on should be in a project with a `pyrefly.toml` configuration

## Step-by-Step Process

### Step 1: Remove ALL Type Suppression Directives

Remove **both** pyre and mypy suppression comments at the top of the file:

```python
# REMOVE lines like these:
# pyre-ignore-all-errors
# pyre-ignore-all-errors[16,21,53,56]
# @lint-ignore-every PYRELINT
# mypy: allow-untyped-defs
# mypy: ignore-errors
# mypy: allow-untyped-decorators
# mypy: allow-untyped-calls
```

These directives suppress type checking for the entire file and must be removed to enable proper type coverage. Removing `# mypy: allow-untyped-defs` is critical — leaving it means mypy won't check the types you're adding.

### Step 2: Add Entry to pyrefly.toml

Add a sub-config entry for stricter type checking. Open `pyrefly.toml` and add an entry following this pattern:

```toml
[[sub-config]]
matches = "path/to/directory/**"
[sub-config.errors]
implicit-import = false
implicit-any = true
bad-param-name-override = false
unannotated-return = true
unannotated-parameter = true
```

**IMPORTANT**: Sub-configs reset the parent error settings. If the global config has
`bad-param-name-override = false`, you must explicitly set it in the sub-config too,
otherwise it defaults to `true` and will surface errors that the global config suppresses.

### Step 3: Run Pyrefly to Identify Missing Coverage

Execute the type checker to see all type errors:

```bash
pyrefly check <FILENAME>
```

Example:
```bash
pyrefly check torch/_dynamo/utils.py
```

This will output a list of type errors with line numbers and descriptions. Common error types include:
- `unannotated-return` — Missing return type annotations
- `unannotated-parameter` — Missing parameter type annotations
- `implicit-any` — Bare generic types like `Callable`, `dict`, `list` without type parameters
- `bad-argument-type` — Incompatible types
- `missing-attribute` — Missing attribute definitions

**CRITICAL**: Your goal is to resolve all `unannotated-return`, `unannotated-parameter`,
and `implicit-any` errors. Other error types (like `bad-argument-type`) are real type bugs
that may require code changes beyond annotations. If you cannot resolve an error, you can
use `# pyrefly: ignore[...]` to suppress but you should try to resolve the error first.

### Step 4: Add Type Annotations

Work through each error systematically:

1. **Read the function/code carefully** - Understand what the function does
2. **Examine usage patterns** - Look at how the function is called to understand expected types
3. **Add appropriate annotations** - Add type hints based on your analysis

#### Backward Compatibility

**CRITICAL**: Functions decorated with `@compatibility(is_backward_compatible=True)` must
NOT have their signatures changed, as this will break the backward compatibility test
(`test_function_back_compat`). Instead, use pyrefly ignore comments:

```python
@compatibility(is_backward_compatible=True)
def my_function(  # pyrefly: ignore[unannotated-return]
    self,
    arg1,  # can't add type here either
):
    ...
```

The `# pyrefly: ignore` comment must be on the `def` line (where pyrefly reports the error),
not on the closing `)`.

#### Common Annotation Patterns

**Function signatures:**
```python
# Before
def process_data(items, callback):
    ...

# After
from collections.abc import Callable
def process_data(items: list[str], callback: Callable[[str], bool]) -> None:
    ...
```

**Class attributes:**
```python
# Before
class MyClass:
    def __init__(self):
        self.value = None
        self.items = []

# After
class MyClass:
    value: int | None
    items: list[str]

    def __init__(self) -> None:
        self.value = None
        self.items = []
```

**Complex types:**
**CRITICAL**: use syntax for Python >3.10 and prefer collections.abc as opposed to
typing for better code standards.

**Critical**: For more advanced/generic types such as `TypeAlias`, `TypeVar`, `Generic`, `Protocol`, etc. use `typing_extensions`

```python
# Optional values
def get_value(key: str) -> int | None: ...

# Union types
def process(value: str | int) -> str: ...

# Dict and List
def transform(data: dict[str, list[int]]) -> list[str]: ...

# Callable — always parameterize, never use bare Callable
from collections.abc import Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int: ...

# When the callable signature is unknown, use Callable[..., Any]
def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]: ...

# Sequence for parameters that accept both list and tuple
from collections.abc import Sequence
def process(items: Sequence[object]) -> None: ...

# TypeVar for generics
from typing_extensions import TypeVar
T = TypeVar('T')
def first(items: list[T]) -> T: ...

# Context managers decorated with @contextmanager
from collections.abc import Generator
@contextmanager
def my_ctx() -> Generator[MyType, None, None]:
    yield value
```

**Empty containers — avoid the `dict[K,V]()` / `list[T]()` antipattern:**

```python
# BAD — creates a GenericAlias at runtime, slower than {}
my_dict = dict[str, Any]()

# GOOD — use a type annotation with a literal
my_dict: dict[str, Any] = {}

# For return statements where you can't annotate, use pyrefly ignore
return [], counter  # pyrefly: ignore[implicit-any]
#                     ^ comment AFTER counter, not before it!
```

**CRITICAL**: When suppressing with `# pyrefly: ignore[...]` on a line with multiple
expressions, make sure the comment goes at the END of the line. Putting it in the middle
will comment out the rest of the line:

```python
# BAD — counter becomes part of the comment!
return []  # pyrefly: ignore[implicit-any], counter

# GOOD
return [], counter  # pyrefly: ignore[implicit-any]
```

**Using `# pyrefly: ignore` for specific lines:**

If a specific line is difficult to type correctly (e.g., dynamic metaprogramming), you can ignore just that line:

```python
# pyrefly: ignore[attr-defined]
result = getattr(obj, dynamic_name)()
```

**CRITICAL**: Avoid using `# pyrefly: ignore` unless it is necessary.
When possible, we can implement stubs, or refactor code to make it more type-safe.

### Step 5: Iterate and Verify

After adding annotations:

1. **Re-run pyrefly check** to verify errors are resolved:
   ```bash
   pyrefly check <FILENAME>
   ```

2. **Fix any new errors** that may appear from the annotations you added.
   Adding a return type annotation can reveal `bad-return` errors where the function
   returns an incompatible type.

3. **Repeat until clean** - Continue until pyrefly reports no target errors

### Step 6: Run Linter

Before committing, run `lintrunner -a` on all changed files to auto-fix formatting:

```bash
lintrunner -a <files...>
```

This will fix import sorting, line length, and other style issues introduced by
the type annotations.

### Step 7: Run Tests

Run the backward compatibility test to ensure no public API signatures changed:

```bash
python -m pytest test/test_fx.py::TestFXAPIBackwardCompatibility -x -v
```

Also run relevant unit tests for the files you modified.

### Step 8: Commit Changes

To keep type coverage PRs manageable, you should commit your change once finished
with a file or a logical group of files.

## Tips for Success

1. **Start with function signatures** - Return types and parameter types are the highest priority

2. **Use `from __future__ import annotations`** - Add this at the top of the file for forward references:
   ```python
   from __future__ import annotations
   ```

3. **Leverage type inference** - Pyrefly can infer many types; focus on function boundaries

4. **Check existing type stubs** - For external libraries, check if type stubs exist

5. **Use `typing_extensions` for newer features** - For compatibility:
   ```python
   from typing_extensions import TypeAlias, Self, ParamSpec
   ```

6. **Watch for forward references in class bodies** - Use string literals for types
   not yet defined:
   ```python
   class MyClass:
       def __new__(cls) -> "MyClass":  # string because class isn't fully defined yet
           ...
   ```

7. **Don't use `dict[K,V]()` or `list[T]()` constructors** - These create a
   `types.GenericAlias` at runtime which is slower than `{}` or `[]`. Use variable
   annotations instead.

8. **When delegating to subagents** - Never let subagents run concurrent git operations
   (checkout, commit, stash). Have them only edit files, then handle git yourself
   sequentially.

## Example Workflow

```bash
# 1. Open the file and remove pyre-ignore-all-errors AND mypy: allow-untyped-defs
# 2. Add entry to pyrefly.toml (with bad-param-name-override = false!)

# 3. Check initial errors
pyrefly check torch/my_module.py

# 4. Add annotations iteratively

# 5. Re-check after changes
pyrefly check torch/my_module.py

# 6. Run linter
lintrunner -a torch/my_module.py

# 7. Run backward compat test
python -m pytest test/test_fx.py::TestFXAPIBackwardCompatibility -x -v

# 8. Repeat until clean, then commit
```
