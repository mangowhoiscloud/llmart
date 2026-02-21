"""LLMART CLI entry point."""

import typer
from rich.console import Console

app = typer.Typer(
    name="llmart",
    help="LLMART — Game Selection & Value Inference Pipeline",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    top_k: int = typer.Option(30, "--top-k", "-k", help="Number of final selections"),
) -> None:
    """Run the full LLMART pipeline."""
    console.print(f"[bold green]LLMART Pipeline[/] — selecting top {top_k} games")


@app.command()
def version() -> None:
    """Show version."""
    from llmart import __version__

    console.print(f"llmart {__version__}")
