"""Executable architecture rules.

Clean architecture degrades quietly: one convenient ``from sqlalchemy import select`` in a
service, and the layering claim in the report stops being true. These tests parse the AST
of every module in the inner layers and fail the build if the dependency rule is violated,
so the boundary is enforced rather than merely documented.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1] / "app"

# Packages the inner layers must never reach for.
FORBIDDEN_BY_LAYER: dict[str, set[str]] = {
    "domain": {"sqlalchemy", "fastapi", "starlette", "redis", "torch", "httpx", "alembic"},
    "application": {"sqlalchemy", "fastapi", "starlette", "redis", "torch", "alembic"},
}

# Inner layers may not import outward.
FORBIDDEN_INTERNAL_BY_LAYER: dict[str, set[str]] = {
    "domain": {"app.application", "app.infrastructure", "app.api"},
    "application": {"app.infrastructure", "app.api"},
}


def _iter_modules(layer: str) -> list[Path]:
    return sorted((APP_ROOT / layer).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level package of every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_app_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names if a.name.startswith("app."))
    return modules


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_BY_LAYER))
def test_layer_does_not_import_frameworks(layer: str) -> None:
    forbidden = FORBIDDEN_BY_LAYER[layer]
    violations: list[str] = []

    for module in _iter_modules(layer):
        offending = _imported_roots(module) & forbidden
        if offending:
            violations.append(f"{module.relative_to(APP_ROOT)} imports {sorted(offending)}")

    assert not violations, "Dependency rule violated:\n" + "\n".join(violations)


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_INTERNAL_BY_LAYER))
def test_layer_does_not_import_outward(layer: str) -> None:
    forbidden = FORBIDDEN_INTERNAL_BY_LAYER[layer]
    violations: list[str] = []

    for module in _iter_modules(layer):
        for imported in _imported_app_modules(module):
            if any(imported.startswith(prefix) for prefix in forbidden):
                violations.append(f"{module.relative_to(APP_ROOT)} imports {imported}")

    assert not violations, "Inward-dependency rule violated:\n" + "\n".join(violations)


def test_composition_root_is_the_only_place_binding_concrete_repositories() -> None:
    """Concrete infrastructure classes must be named in ``api/deps.py``, the tests, or
    other infrastructure — never in a router, where it would bypass injection."""
    offenders: list[str] = []

    for module in sorted((APP_ROOT / "api").rglob("*.py")):
        if module.name == "deps.py":
            continue
        for imported in _imported_app_modules(module):
            if imported.startswith("app.infrastructure"):
                offenders.append(f"{module.relative_to(APP_ROOT)} imports {imported}")

    assert not offenders, "Routers must receive dependencies, not construct them:\n" + "\n".join(
        offenders
    )


def test_every_domain_port_is_a_protocol() -> None:
    """Guards against a port silently becoming a concrete base class, which would force
    inheritance on adapters and defeat structural typing."""
    source = (APP_ROOT / "domain" / "interfaces.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes, "no ports declared"

    for node in classes:
        base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
        assert "Protocol" in base_names, f"{node.name} is not a Protocol"
