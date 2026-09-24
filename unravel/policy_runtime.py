"""Prepare this process for the policy app's HTTP clients."""

import os


def configure_policy_network() -> None:
    """Use the configured HTTP proxy when ALL_PROXY has an invalid SOCKS scheme."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    if os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"):
        for name in ("ALL_PROXY", "all_proxy"):
            if os.getenv(name, "").startswith("socks://"):
                os.environ.pop(name, None)
