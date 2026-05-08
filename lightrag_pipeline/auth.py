import logging
import os
from functools import wraps

from flask import g, jsonify, request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]


def verify_token(token: str) -> dict:
    idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
    return idinfo


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            g.claims = verify_token(auth.split(" ", 1)[1])
            g.user_email = g.claims.get("email", "").lower()
        except (ValueError, Exception) as e:
            logger.warning(f"Auth failed: {e}")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper
