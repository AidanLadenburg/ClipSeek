"""eMAM REST API client: local config/token persistence + proxy UUID -> full-res path lookup."""

import json
import os
import time
import traceback
from typing import Optional, TypedDict

import requests

TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS = 300
TOKEN_REQUEST_TIMEOUT_SECONDS = 10
ASSET_REQUEST_TIMEOUT_SECONDS = 15


class EmamError(Exception):
    pass


class EmamConfig(TypedDict, total=False):
    enabled: bool
    ip: str
    username: str
    password: str
    license_key: str


def _strip_scheme(ip: str) -> str:
    working = str(ip or "").strip()
    for prefix in ("http://", "https://"):
        if working.lower().startswith(prefix):
            working = working[len(prefix):]
    return working.rstrip("/")


class EmamClient:
    def __init__(self, config_dir: Optional[str] = None) -> None:
        self._config_dir = config_dir or os.path.join(os.path.expanduser("~"), ".clipseek")
        self._config_path = os.path.join(self._config_dir, "emam_config.json")
        self._token_cache_path = os.path.join(self._config_dir, "emam_token_cache.json")

        self._config: EmamConfig = self._load_config()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._load_cached_token()

    # ---- config persistence ----

    def _load_config(self) -> EmamConfig:
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_config(self, new_config: dict) -> None:
        merged = dict(self._config)
        credential_keys = ("ip", "username", "password", "license_key")
        credentials_changed = False
        for key in ("enabled",) + credential_keys:
            if key not in new_config:
                continue
            if key == "password" and not new_config.get("password"):
                # Blank password on save means "keep the existing one".
                continue
            if key in credential_keys and new_config.get(key) != merged.get(key):
                credentials_changed = True
            merged[key] = new_config[key]

        os.makedirs(self._config_dir, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        self._config = merged

        if credentials_changed:
            self._invalidate_token()

    def get_config_for_display(self) -> dict:
        return {
            "enabled": bool(self._config.get("enabled")),
            "ip": self._config.get("ip", ""),
            "username": self._config.get("username", ""),
            "license_key": self._config.get("license_key", ""),
            "has_password": bool(self._config.get("password")),
        }

    def is_enabled(self) -> bool:
        return bool(self._config.get("enabled"))

    # ---- token cache ----

    def _load_cached_token(self) -> None:
        try:
            with open(self._token_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._token = data.get("access_token")
            self._token_expires_at = float(data.get("expires_at", 0))
        except Exception:
            self._token = None
            self._token_expires_at = 0.0

    def _save_cached_token(self) -> None:
        try:
            os.makedirs(self._config_dir, exist_ok=True)
            with open(self._token_cache_path, "w", encoding="utf-8") as f:
                json.dump({"access_token": self._token, "expires_at": self._token_expires_at}, f)
        except Exception:
            pass

    def _invalidate_token(self) -> None:
        self._token = None
        self._token_expires_at = 0.0
        try:
            os.remove(self._token_cache_path)
        except OSError:
            pass

    def _token_is_valid(self) -> bool:
        return bool(self._token) and time.time() < (
            self._token_expires_at - TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS
        )

    def _fetch_token(self) -> None:
        ip = _strip_scheme(self._config.get("ip", ""))
        url = f"http://{ip}/emamrestapi/token"
        data = {
            "grant_type": "password",
            "client_id": "0",
            "client_secret": self._config.get("license_key", ""),
            "username": self._config.get("username", ""),
            "password": self._config.get("password", ""),
        }
        try:
            resp = requests.post(url, data=data, timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise EmamError(f"Could not reach eMAM server for token: {e}") from e

        if resp.status_code != 200:
            raise EmamError(f"eMAM token request failed ({resp.status_code}): {resp.text[:200]}")

        try:
            payload = resp.json()
        except ValueError as e:
            raise EmamError(f"eMAM token response was not valid JSON: {e}") from e

        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not token or not isinstance(expires_in, (int, float)):
            raise EmamError("eMAM token response missing access_token/expires_in")

        self._token = token
        self._token_expires_at = time.time() + expires_in
        self._save_cached_token()

    def _ensure_token(self) -> str:
        if not all(
            self._config.get(k)
            for k in ("ip", "username", "password", "license_key")
        ):
            raise EmamError("EMAM not configured")
        if not self._token_is_valid():
            self._fetch_token()
        return self._token

    # ---- asset lookup ----

    def resolve_uuid(self, uuid: str) -> str:
        if not uuid:
            raise EmamError("No UUID provided")

        token = self._ensure_token()
        ip = _strip_scheme(self._config.get("ip", ""))
        url = f"http://{ip}/eMAMRESTAPI/api/v1/assets"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        params = {"UUID": uuid}

        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=ASSET_REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as e:
            raise EmamError(f"Could not reach eMAM server for asset lookup: {e}") from e

        if resp.status_code == 401:
            # Token may have been invalidated server-side; force a refresh and retry once.
            self._invalidate_token()
            token = self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            try:
                resp = requests.get(
                    url, headers=headers, params=params, timeout=ASSET_REQUEST_TIMEOUT_SECONDS
                )
            except requests.RequestException as e:
                raise EmamError(f"Could not reach eMAM server for asset lookup: {e}") from e

        if resp.status_code != 200:
            raise EmamError(f"eMAM asset lookup failed ({resp.status_code}): {resp.text[:200]}")

        try:
            payload = resp.json()
        except ValueError as e:
            raise EmamError(f"eMAM asset response was not valid JSON: {e}") from e

        results = payload.get("results") or []
        if not results:
            raise EmamError(f"No eMAM asset found for UUID {uuid}")

        face_version = results[0].get("faceVersion") or {}
        original_source_path = face_version.get("originalSourcePath")
        file_name = face_version.get("fileName")
        if not original_source_path or not file_name:
            raise EmamError("eMAM asset response missing faceVersion path/fileName")

        return original_source_path.rstrip("\\") + "\\" + file_name
