"""Argon2id password hashing.

Argon2id is the current OWASP recommendation: it is memory-hard, which raises the cost of
GPU and ASIC cracking in a way that PBKDF2 and bcrypt cannot match. Cost parameters are
constructor arguments so they can be raised over the project's life, with
``needs_rehash`` upgrading existing users transparently at their next successful login.
"""

from __future__ import annotations

import contextlib

from argon2 import PasswordHasher as Argon2Hasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.logging import get_logger
from app.domain.interfaces import PasswordHasher

logger = get_logger(__name__)


class Argon2PasswordHasher(PasswordHasher):
    """Argon2id implementation of the :class:`PasswordHasher` port."""

    def __init__(
        self,
        *,
        time_cost: int = 3,
        memory_cost_kib: int = 65536,  # 64 MiB
        parallelism: int = 4,
        hash_length: int = 32,
        salt_length: int = 16,
    ) -> None:
        self._hasher = Argon2Hasher(
            time_cost=time_cost,
            memory_cost=memory_cost_kib,
            parallelism=parallelism,
            hash_len=hash_length,
            salt_len=salt_length,
        )
        # Precomputed once so that dummy_verify() costs the same as a real verification
        # without paying a hashing cost to produce the target each time.
        self._dummy_hash = self._hasher.hash("geovision-timing-equalisation-placeholder")

    def hash(self, plain_password: str) -> str:
        return self._hasher.hash(plain_password)

    def verify(self, plain_password: str, password_hash: str) -> bool:
        try:
            return self._hasher.verify(password_hash, plain_password)
        except VerifyMismatchError:
            return False
        except (VerificationError, InvalidHashError):
            # A corrupt or foreign-format hash is a data problem, not a user error;
            # surface it in logs but still deny access.
            logger.warning("password_hash_unverifiable")
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            return True

    def dummy_verify(self) -> None:
        """Spend a verification's worth of CPU against a throwaway hash.

        Called when no user matches the submitted email so that login latency does not
        distinguish "unknown account" from "wrong password".
        """
        with contextlib.suppress(VerifyMismatchError, VerificationError, InvalidHashError):
            self._hasher.verify(self._dummy_hash, "not-the-password")
