"""Command-line entry point for downloading the public Shanghai law corpus."""

from pathlib import Path

import click

from unravel.services.policy_source import (
    DEFAULT_SOURCE_PATH,
    download_official_policy_corpus,
)


@click.command()
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SOURCE_PATH,
    show_default=True,
    help="Path for the downloaded policy JSON.",
)
@click.option("--workers", type=click.IntRange(min=1, max=12), default=12, show_default=True)
@click.option(
    "--request-interval",
    type=click.FloatRange(min=0.0),
    default=0.05,
    show_default=True,
    help="Minimum seconds between detail requests.",
)
def main(output_path: Path, workers: int, request_interval: float) -> None:
    """Download current records and full text from Shanghai's official law portal."""
    result = download_official_policy_corpus(
        output_path,
        workers=workers,
        minimum_interval=request_interval,
    )
    click.echo(f"Downloaded: {result['downloaded_count']}/{result['source_count']}")
    click.echo(f"Status totals: {result['status_counts']}")
    click.echo(f"Records with an empty full text: {result['empty_body_count']}")
    click.echo(f"JSON file: {result['output_path']}")


if __name__ == "__main__":
    main()
