"""Standalone Streamlit entry point for the Shanghai policy walkthrough."""

import streamlit as st

from unravel.policy_runtime import configure_policy_network


def main() -> None:
    """Render the policy page using the existing OpenCode credential."""
    configure_policy_network()

    from unravel.cli import _load_deepseek_key_from_opencode
    from unravel.ui.steps.policy import render_policy_step
    from unravel.utils.ui import apply_custom_styles

    st.set_page_config(page_title="Shanghai Policy RAG", page_icon="U", layout="wide")
    _load_deepseek_key_from_opencode()
    apply_custom_styles()
    render_policy_step()


if __name__ == "__main__":
    main()
