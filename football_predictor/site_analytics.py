from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class SiteAnalyticsError(RuntimeError):
    """Ошибка подключения или чтения статистики посещений."""


@dataclass(frozen=True)
class AnalyticsSummary:
    online_now: int = 0
    visits_today: int = 0
    total_visits: int = 0


class SiteAnalyticsClient:
    """Небольшой клиент Supabase REST API без дополнительной зависимости."""

    def __init__(
        self,
        url: str = "",
        publishable_key: str = "",
        secret_key: str = "",
        timeout: float = 5.0,
    ) -> None:
        self.url = str(url or "").rstrip("/")
        self.publishable_key = str(publishable_key or "").strip()
        self.secret_key = str(secret_key or "").strip()
        self.timeout = float(timeout)

    @property
    def public_enabled(self) -> bool:
        return bool(self.url and self.publishable_key)

    @property
    def admin_enabled(self) -> bool:
        return bool(self.url and self.secret_key)

    def _headers(self, key: str) -> dict[str, str]:
        headers = {
            "apikey": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "football-predictor-site-analytics/4.6",
        }
        # Legacy anon/service_role keys are JWTs. New sb_publishable/sb_secret
        # keys are sent through apikey only, in accordance with Supabase docs.
        if key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _rpc(self, function_name: str, payload: dict[str, Any] | None = None) -> Any:
        if not self.public_enabled:
            raise SiteAnalyticsError("Публичный ключ аналитики не настроен.")
        response = requests.post(
            f"{self.url}/rest/v1/rpc/{function_name}",
            headers=self._headers(self.publishable_key),
            json=payload or {},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise SiteAnalyticsError(
                f"Supabase RPC {function_name}: HTTP {response.status_code}: {response.text[:300]}"
            )
        if not response.content:
            return None
        return response.json()

    def register_session(self, session_id: str, page: str = "main") -> None:
        self._rpc(
            "register_site_session",
            {"p_session_id": session_id, "p_page": page},
        )

    def heartbeat(self, session_id: str, page: str = "main") -> None:
        self._rpc(
            "heartbeat_site_session",
            {"p_session_id": session_id, "p_page": page},
        )

    def public_summary(self) -> AnalyticsSummary:
        data = self._rpc("get_site_public_summary")
        if isinstance(data, list) and data:
            row = data[0]
        elif isinstance(data, dict):
            row = data
        else:
            row = {}
        return AnalyticsSummary(
            online_now=int(row.get("online_now") or 0),
            visits_today=int(row.get("visits_today") or 0),
            total_visits=int(row.get("total_visits") or 0),
        )

    def fetch_sessions(self, max_rows: int = 20_000) -> list[dict[str, Any]]:
        if not self.admin_enabled:
            raise SiteAnalyticsError("Секретный серверный ключ аналитики не настроен.")

        rows: list[dict[str, Any]] = []
        page_size = 1000
        offset = 0
        endpoint = f"{self.url}/rest/v1/site_sessions"
        params_base = {
            "select": "session_id,first_seen,last_seen,first_page,last_page",
            "order": "first_seen.desc",
        }
        while offset < max_rows:
            params = dict(params_base)
            params["limit"] = str(min(page_size, max_rows - offset))
            params["offset"] = str(offset)
            response = requests.get(
                endpoint,
                headers=self._headers(self.secret_key),
                params=params,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise SiteAnalyticsError(
                    f"Supabase history: HTTP {response.status_code}: {response.text[:300]}"
                )
            batch = response.json()
            if not isinstance(batch, list):
                raise SiteAnalyticsError("Supabase вернул историю в неожиданном формате.")
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return rows
