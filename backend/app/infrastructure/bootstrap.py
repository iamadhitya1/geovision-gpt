"""First-run bootstrapping.

Creates the initial administrator so a fresh deployment is usable without hand-editing the
database. Idempotent — safe to run on every startup and across replicas.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.password_policy import PasswordPolicy
from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.entities import User
from app.domain.enums import Role
from app.domain.interfaces import PasswordHasher
from app.infrastructure.db.repositories import SqlAlchemyUserRepository

logger = get_logger(__name__)

_INSECURE_DEFAULT_PASSWORD = "ChangeMe_Please_2026"


async def ensure_first_admin(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    hasher: PasswordHasher,
) -> None:
    """Create the bootstrap administrator when no account holds that email.

    Refuses to run in production with the shipped default password — a deployment that
    silently creates a publicly-known admin credential is worse than one that fails to
    start. The password is also checked against the same :class:`PasswordPolicy` every
    other account is held to; without this, a weak ``GVG_FIRST_ADMIN_PASSWORD`` would
    silently create the platform's highest-privilege account with a password that
    ``POST /auth/register`` would reject outright for anyone else.
    """
    email = settings.first_admin_email.lower()
    password = settings.first_admin_password.get_secret_value()

    if settings.is_production and password == _INSECURE_DEFAULT_PASSWORD:
        raise RuntimeError(
            "Refusing to bootstrap the admin account with the default password in "
            "production. Set GVG_FIRST_ADMIN_PASSWORD."
        )

    session = session_factory()
    try:
        repository = SqlAlchemyUserRepository(session)
        if await repository.exists_by_email(email):
            logger.debug("bootstrap_admin_exists", email=email)
            return

        # Validated only on the create path: a policy tightened after an admin already
        # exists must not turn every subsequent restart into a startup failure for a
        # password that was acceptable when the account was actually created.
        PasswordPolicy(min_length=settings.password_min_length).validate(password, email=email)

        admin = User(
            email=email,
            full_name=settings.first_admin_name,
            password_hash=hasher.hash(password),
            role=Role.ADMIN,
            is_verified=True,
        )
        await repository.add(admin)
        await session.commit()
        logger.info("bootstrap_admin_created", email=email)
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
