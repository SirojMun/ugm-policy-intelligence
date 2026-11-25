"""Polite, robots.txt-compliant HTTP client with rate limiting, retries, and .ac.id SSL fallback."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import certifi
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class HttpConfig:
    """Configuration for ethical crawling and robust HTTP requests."""

    user_agent: str
    min_delay_sec: float = 1.5
    max_delay_sec: float = 3.0
    timeout_sec: int = 25
    max_retries: int = 3
    backoff_base_sec: float = 2.0
    allow_insecure_ssl_fallback: bool = True
    insecure_ssl_hosts: tuple[str, ...] = ("sa.ugm.ac.id", "senatakademik.ugm.ac.id", "jdihn.go.id")
    sso_indicators: tuple[str, ...] = (
        "sso.ugm.ac.id",
        "/ugmfw/signin",
        "/spse/signin",
        "/wp-login.php",
        "accounts.google.com",
        "/login",
        "/signin",
    )


class EthicalHttpClient:
    """HTTP client enforcing robots.txt, polite random delays, backoff, and SSO checks."""

    def __init__(self, config: HttpConfig, disable_delay: bool = False) -> None:
        self.config = config
        self.disable_delay = disable_delay
        self._last_request_at: float = 0.0
        self._robots_cache: dict[str, RobotFileParser | None] = {}
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": self.config.user_agent})
        retry = Retry(
            total=self.config.max_retries,
            read=self.config.max_retries,
            connect=self.config.max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            backoff_factor=self.config.backoff_base_sec,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def is_sso_url(self, url: str) -> bool:
        """Return True if URL clearly points to an SSO/login gate."""
        lowered = (url or "").lower()
        return any(ind in lowered for ind in self.config.sso_indicators)

    def _polite_sleep(self) -> None:
        if self.disable_delay:
            return
        elapsed = time.monotonic() - self._last_request_at
        target = random.uniform(self.config.min_delay_sec, self.config.max_delay_sec)
        if elapsed < target and self._last_request_at > 0:
            time.sleep(target - elapsed)

    def can_fetch(self, url: str) -> bool:
        """Check robots.txt for the URL's host. Caches robots.txt per scheme+netloc."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        if self.is_sso_url(url):
            return False

        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots_cache:
            self._robots_cache[origin] = self._load_robots(origin, parsed.netloc)

        rp = self._robots_cache[origin]
        if rp is None:
            return True
        return rp.can_fetch(self.config.user_agent, url)

    def _load_robots(self, origin: str, netloc: str) -> RobotFileParser | None:
        robots_url = f"{origin}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            resp = self._raw_get(robots_url, allow_redirects=True, host_hint=netloc)
            if resp.status_code == 200 and resp.text:
                # Guard against HTML error/landing pages masquerading as robots.txt
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "html" in ct or "<html" in resp.text[:300].lower():
                    logger.info("robots.txt for %s returned HTML; treating as permissive.", origin)
                    return None
                rp.parse(resp.text.splitlines())
                logger.info("Loaded robots.txt from %s", robots_url)
                return rp
            logger.info("No active robots.txt at %s (status=%s); proceeding politely.", robots_url, resp.status_code)
            return None
        except Exception as exc:
            logger.warning("Failed to fetch robots.txt for %s (%s); treating as permissive.", origin, exc)
            return None

    def _raw_get(self, url: str, allow_redirects: bool = True, host_hint: str | None = None, stream: bool = False) -> requests.Response:
        self._polite_sleep()
        host = (host_hint or urlparse(url).netloc).lower()
        try:
            resp = self.session.get(
                url,
                timeout=self.config.timeout_sec,
                allow_redirects=allow_redirects,
                verify=certifi.where(),
                stream=stream,
            )
            self._last_request_at = time.monotonic()
            return resp
        except requests.exceptions.SSLError as ssl_err:
            can_fallback = self.config.allow_insecure_ssl_fallback and (
                host in self.config.insecure_ssl_hosts or host.endswith(".ac.id")
            )
            if not can_fallback:
                raise
            logger.warning("SSL verification failed for %s (%s). Retrying with verify=False (explicit .ac.id fallback).", url, ssl_err)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            self._polite_sleep()
            resp = self.session.get(
                url,
                timeout=self.config.timeout_sec,
                allow_redirects=allow_redirects,
                verify=False,
                stream=stream,
            )
            self._last_request_at = time.monotonic()
            return resp

    def get(self, url: str, check_robots: bool = True, stream: bool = False) -> requests.Response:
        """Perform a polite, robots-checked GET request and validate no SSO redirect."""
        if check_robots and not self.can_fetch(url):
            raise PermissionError(f"Blocked by robots.txt or SSO policy: {url}")

        resp = self._raw_get(url, allow_redirects=True, stream=stream)
        if self.is_sso_url(resp.url):
            resp.close()
            raise PermissionError(f"Redirected to SSO/login gate: {resp.url}")
        if resp.status_code in (401, 403):
            # Some WP sites return 403 for login-only files
            raise PermissionError(f"Access denied (HTTP {resp.status_code}) for {url}")
        resp.raise_for_status()
        return resp
