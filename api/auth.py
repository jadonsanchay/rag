"""Signup/login/logout/me — session-based auth.

Sessions, not JWT: a random token in an `sessions` table (pipeline/registry.py),
handed to the browser as an HttpOnly cookie. Logout deletes the row, which is
real, immediate revocation — the thing a stateless JWT can't do without a
blocklist. No CSRF token: the cookie is HttpOnly + SameSite=Lax, and every
mutating request here needs a JSON body, which a cross-site HTML <form> can't
send — that combination is a standard, sufficient mitigation for this threat
model, not an oversight.
"""

import os
import re
import secrets
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.schemas import LoginRequest, SignupRequest, UserOut
from pipeline.registry import User, get_registry

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "session_token"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
# Local dev is plain HTTP, where a Secure cookie is silently dropped; fly.toml
# sets this true for the deployed instance (which is HTTPS-only).
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Precomputed so a login against a nonexistent email costs roughly the same
# time as one against a real email with the wrong password — bcrypt only runs
# when a user is found, and that time difference alone would let someone
# enumerate registered emails.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing-parity", bcrypt.gensalt())


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        path="/",
    )


def get_current_user(request: Request) -> Optional[User]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return get_registry().get_session_user(token)


def require_user(request: Request) -> User:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _to_user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email)


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(request: SignupRequest, response: Response) -> UserOut:
    email = request.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address")

    registry = get_registry()
    if registry.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = registry.create_user(email, _hash_password(request.password))
    token = secrets.token_urlsafe(32)
    registry.create_session(user.id, token, SESSION_TTL_SECONDS)
    _set_session_cookie(response, token)
    return _to_user_out(user)


@router.post("/login", response_model=UserOut)
def login(request: LoginRequest, response: Response) -> UserOut:
    email = request.email.strip().lower()
    registry = get_registry()
    user = registry.get_user_by_email(email)

    if user is None:
        bcrypt.checkpw(request.password.encode("utf-8"), _DUMMY_HASH)
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not _verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = secrets.token_urlsafe(32)
    registry.create_session(user.id, token, SESSION_TTL_SECONDS)
    _set_session_cookie(response, token)
    return _to_user_out(user)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        get_registry().delete_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(require_user)) -> UserOut:
    return _to_user_out(user)
