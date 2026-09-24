"""CLI entry point for Unravel."""

import json
import os
import sys
from pathlib import Path

import click


def _load_deepseek_key_from_opencode() -> None:
    """Expose OpenCode's saved DeepSeek key to this process without persisting it."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return

    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    entry = payload.get("deepseek") if isinstance(payload, dict) else None
    if not isinstance(entry, dict) or entry.get("type") != "api":
        return

    api_key = entry.get("key")
    if isinstance(api_key, str) and api_key.strip():
        os.environ["DEEPSEEK_API_KEY"] = api_key.strip()


@click.command()
@click.option(
    "--port",
    "-p",
    default=8501,
    type=int,
    help="Port to run the Streamlit app on.",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    type=str,
    help="Host to bind the Streamlit app to.",
)
@click.option(
    "--app",
    type=click.Choice(["finance", "playground"]),
    default="finance",
    show_default=True,
    help="Choose the financial report Q&A or the original RAG playground.",
)
@click.version_option(package_name="unravel")
def main(port: int, host: str, app: str) -> None:
    """Launch the financial report Q&A or the original Unravel playground."""
    # Import here to avoid slow startup for --help
    from streamlit.web import cli as stcli

    _load_deepseek_key_from_opencode()

    if app == "finance":
        app_path = Path(__file__).resolve().parent / "finance" / "app.py"
        title = "Financial Report RAG"
    else:
        from unravel.services.storage import ensure_storage_dir

        storage_path = ensure_storage_dir()
        click.echo(f"Storage directory: {storage_path}")
        app_path = Path(__file__).resolve().parent / "app.py"
        title = "Unravel RAG Playground"

    click.echo(f"Starting {title} on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the server.\n")

    # Build streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
    ]

    # Launch Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
