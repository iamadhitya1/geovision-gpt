"""Version 1 API router aggregation.

New modules register here. Keeping every include in one file means the full public surface
of v1 is readable at a glance, and versioning to v2 later is a directory copy rather than
an archaeology exercise.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, health, images, users, vqa

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(images.router)
api_router.include_router(vqa.router)

# Module 6 — the frontend, a separate app, not a router here
# Module 7  — api_router.include_router(gis.router)
# Module 8  — api_router.include_router(analytics.router)
