"""Password strength policy.

A small, injectable rule object rather than a scatter of ``if len(pw) < 8`` checks. It is
constructed once in the composition root with values from configuration, so the policy can
be tightened for production without editing service code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.errors import WeakPasswordError

_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password123",
        "12345678",
        "123456789",
        "qwerty123",
        "letmein",
        "admin123",
        "welcome1",
        "iloveyou",
        "changeme",
        "geovision",
    }
)


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    """Validates candidate passwords against a configurable rule set."""

    min_length: int = 10
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_symbol: bool = False

    def validate(self, password: str, *, email: str | None = None) -> None:
        """Raise :class:`WeakPasswordError` when ``password`` violates the policy.

        All violations are collected before raising so the user sees every problem at
        once instead of fixing them one round trip at a time.
        """
        violations: list[str] = []

        if len(password) < self.min_length:
            violations.append(f"must be at least {self.min_length} characters")
        if len(password) > self.max_length:
            violations.append(f"must be at most {self.max_length} characters")
        if self.require_uppercase and not re.search(r"[A-Z]", password):
            violations.append("must contain an uppercase letter")
        if self.require_lowercase and not re.search(r"[a-z]", password):
            violations.append("must contain a lowercase letter")
        if self.require_digit and not re.search(r"\d", password):
            violations.append("must contain a digit")
        if self.require_symbol and not re.search(r"[^A-Za-z0-9]", password):
            violations.append("must contain a symbol")
        if password.lower() in _COMMON_PASSWORDS:
            violations.append("is too common")
        if email:
            local_part = email.split("@", 1)[0].lower()
            if len(local_part) >= 3 and local_part in password.lower():
                violations.append("must not contain your email address")

        if violations:
            raise WeakPasswordError(
                "Password " + "; ".join(violations) + ".",
                violations=violations,
            )
