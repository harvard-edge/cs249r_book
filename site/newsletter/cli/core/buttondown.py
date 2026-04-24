"""Thin Buttondown REST API client.

Deliberately small: only the endpoints the CLI needs (images, emails).
Does not model the whole API. If the API changes, this is the one place
to update.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.buttondown.com/v1"
DEFAULT_TIMEOUT_SECS = 30
UPLOAD_TIMEOUT_SECS = 60

# Map file extensions to MIME types for image uploads.
IMAGE_MIME_TYPES: dict[str, str] = {
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}

logger = logging.getLogger(__name__)


class ButtondownError(RuntimeError):
    """Raised when the Buttondown API returns an error."""


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Token {api_key}"}


def upload_image(api_key: str, image_path: Path) -> str:
    """Upload an image. Returns the CDN URL assigned by Buttondown.

    Raises:
        ButtondownError: on upload failure or unsupported file type.
    """
    if not image_path.exists():
        raise ButtondownError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()
    mime = IMAGE_MIME_TYPES.get(suffix)
    if mime is None:
        raise ButtondownError(f"Unsupported image type: {suffix}")

    logger.debug("Uploading image: %s (%s)", image_path.name, mime)
    with image_path.open("rb") as fp:
        files = {"image": (image_path.name, fp, mime)}
        response = requests.post(
            f"{API_BASE}/images",
            headers=_auth_headers(api_key),
            files=files,
            timeout=UPLOAD_TIMEOUT_SECS,
        )

    if response.status_code >= 400:
        raise ButtondownError(
            f"Image upload failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    url = data.get("image") or data.get("url")
    if not url:
        raise ButtondownError(f"Unexpected image response: {data}")
    return url


def create_draft(
    api_key: str,
    subject: str,
    body: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a draft email. The draft is NOT sent.

    Returns:
        The Buttondown email object, including `id` and `creation_url`.
    """
    payload: dict[str, Any] = {
        "subject": subject,
        "body": body,
        "status": "draft",
    }
    if description:
        payload["description"] = description

    logger.debug("Creating Buttondown draft: %r", subject)
    response = requests.post(
        f"{API_BASE}/emails",
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECS,
    )

    if response.status_code >= 400:
        raise ButtondownError(
            f"Draft creation failed ({response.status_code}): {response.text}"
        )
    return response.json()


def list_emails(api_key: str, status: str | None = None) -> list[dict[str, Any]]:
    """List emails, optionally filtered by status (draft, scheduled, sent).

    Follows Buttondown's pagination (`next` URL) until exhausted so callers
    get the full history, not just the first page.
    """
    params = {"status": status} if status else {}
    logger.debug("Listing Buttondown emails (status=%s)", status)

    url: str | None = f"{API_BASE}/emails"
    results: list[dict[str, Any]] = []
    first_request = True

    while url:
        response = requests.get(
            url,
            headers=_auth_headers(api_key),
            params=params if first_request else None,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
        first_request = False
        if response.status_code >= 400:
            raise ButtondownError(
                f"List failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        if isinstance(data, list):
            results.extend(data)
            url = None
        else:
            results.extend(data.get("results", []))
            url = data.get("next")
    return results


def get_email(api_key: str, email_id: str) -> dict[str, Any]:
    """Fetch a single email by its Buttondown id."""
    logger.debug("Fetching Buttondown email %s", email_id)
    response = requests.get(
        f"{API_BASE}/emails/{email_id}",
        headers=_auth_headers(api_key),
        timeout=DEFAULT_TIMEOUT_SECS,
    )
    if response.status_code >= 400:
        raise ButtondownError(
            f"Get failed ({response.status_code}): {response.text}"
        )
    return response.json()


def update_email(
    api_key: str,
    email_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """PATCH an existing email (partial update).

    Useful for attaching structured data (e.g. ``metadata.author``) to
    already-sent emails without editing their body. The payload is sent
    as-is; common usage::

        update_email(key, eid, {"metadata": {"author": "Marco Zennaro"}})

    Note on ``metadata``: Buttondown replaces the whole dict on PATCH,
    so callers that want to merge should fetch-modify-write.
    """
    logger.debug("Patching Buttondown email %s: keys=%s", email_id, list(payload.keys()))
    response = requests.patch(
        f"{API_BASE}/emails/{email_id}",
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECS,
    )
    if response.status_code >= 400:
        raise ButtondownError(
            f"Update failed ({response.status_code}): {response.text}"
        )
    return response.json()
