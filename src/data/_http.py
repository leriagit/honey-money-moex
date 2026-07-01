"""
HTTP-helper: устойчивый к SSL-проблемам клиент на urllib + certifi.

На macOS Python (особенно brew-сборка и python.org installer без
Install Certificates.command) не подхватывает системные CA-сертификаты.
Это приводит к `SSL: CERTIFICATE_VERIFY_FAILED` при любом HTTPS-запросе.

Этот модуль создаёт SSL-контекст из certifi (bundled CA bundle), который
работает одинаково на macOS, Linux в Docker, и на серверах Yandex Cloud.

Используется в:
  - scripts/check_access.py
  - src/data/arenago_client.py
  - другие места где urllib.request с HTTPS
"""
from __future__ import annotations

import ssl
import urllib.request
from typing import Optional


_ssl_context: Optional[ssl.SSLContext] = None


def get_ssl_context() -> ssl.SSLContext:
    """
    Возвращает кешированный SSL-контекст с certifi CA bundle.
    Если certifi не установлен — fallback на системный контекст.
    """
    global _ssl_context
    if _ssl_context is not None:
        return _ssl_context

    try:
        import certifi
        _ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # certifi не установлен — используем дефолтный контекст
        _ssl_context = ssl.create_default_context()

    return _ssl_context


def urlopen(
    url_or_request,
    *,
    timeout: float = 10.0,
    data: Optional[bytes] = None,
):
    """
    Drop-in замена urllib.request.urlopen — но с надёжным SSL.

    Принимает либо строку-URL, либо Request-объект.
    """
    ctx = get_ssl_context()
    return urllib.request.urlopen(url_or_request, data=data, timeout=timeout, context=ctx)
