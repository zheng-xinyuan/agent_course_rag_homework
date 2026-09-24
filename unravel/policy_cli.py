"""Command-line entry point for the one-time policy index build."""

from pathlib import Path

import click

from unravel.services.policy_corpus import load_policy_corpus
from unravel.services.policy_index import (
    INDEX_DIRECTORY,
    PPT_EXPECTED_CHUNK_COUNT,
    build_policy_index,
)
from unravel.services.policy_source import DEFAULT_SOURCE_PATH
from unravel.policy_runtime import configure_policy_network


@click.command()
@click.argument(
    "source",
    required=False,
    default=DEFAULT_SOURCE_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=INDEX_DIRECTORY,
    show_default=True,
    help="Directory used to store the generated index.",
)
@click.option("--chunk-size", type=click.IntRange(min=100), default=800, show_default=True)
@click.option("--overlap", type=click.IntRange(min=0), default=120, show_default=True)
def main(source: Path, output_dir: Path, chunk_size: int, overlap: int) -> None:
    """Build the Shanghai policy index once before starting the web app."""
    configure_policy_network()
    corpus = load_policy_corpus(source)
    click.echo(f"Policies before filtering: {corpus.source_count}")
    click.echo(f"Status totals: {corpus.status_counts}")
    click.echo(
        "Policy counts: "
        f"current={corpus.category_counts['active']}, "
        f"repealed={corpus.category_counts['repealed']}, "
        f"superseded editions={corpus.category_counts['superseded']}"
    )

    if overlap >= chunk_size:
        raise click.BadParameter("overlap must be smaller than chunk-size")

    manifest = build_policy_index(
        source,
        output_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    click.echo(f"Policies after filtering: {manifest['active_law_count']}")
    click.echo(f"Chunks: {manifest['chunk_count']}")
    click.echo(f"Embedding dimension: {manifest['embedding_dimension']}")
    click.echo(f"Indexing time: {manifest['indexing_seconds']:.2f} seconds")
    click.echo(f"Index directory: {output_dir.expanduser()}")
    if manifest["ppt_status_count_check"] != "match":
        click.echo(
            "Warning: source status counts do not match the PPT's 1,345 / 410 / 92 split."
        )
    if manifest["ppt_chunk_count_check"] != "match":
        click.echo(
            f"Warning: the PPT reports {PPT_EXPECTED_CHUNK_COUNT:,} text chunks; "
            "this chunk configuration produced a different count."
        )


if __name__ == "__main__":
    main()
