#!/usr/bin/env python3
"""
FitScan command-line interface.
Usage: python cli/fitscan_cli.py measure photo.jpg --height 175
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(name="fitscan", help="AI-powered body measurement from photos.")
console = Console()


@app.command()
def measure(
    image: Path = typer.Argument(..., help="Path to photo"),
    height: float | None = typer.Option(None, "--height", "-h", help="Height in cm"),
    reference: float | None = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference object long side in cm",
    ),
    no_smpl: bool = typer.Option(False, "--no-smpl", help="Disable SMPL fitting"),
    no_claude: bool = typer.Option(
        False, "--no-claude", help="Disable Claude validation"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
    gender: str = typer.Option("ladies", "--gender", "-g", help="Gender profile"),
) -> None:
    """Measure a person's body from a photograph."""
    from fitscan.application.measurement_service import (
        MeasurementService,
        create_pipeline,
    )

    if not image.exists():
        console.print(f"[red]File not found: {image}[/red]")
        raise typer.Exit(1)

    with console.status("Running body measurement pipeline..."):
        service = MeasurementService(
            pipeline=create_pipeline(
                enable_smpl=not no_smpl,
                enable_claude=not no_claude,
            )
        )
        result = service.measure(
            image_path=image,
            height_cm=height,
            reference_object_cm=reference,
            gender_profile=gender,
        )

    if output_json:
        print(json.dumps(result, indent=2))
        return

    table = Table(title=f"Body Measurements — {image.name}")
    table.add_column("Measurement", style="cyan")
    table.add_column("Value (cm)", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Source", style="dim")

    for field, data in result.items():
        if isinstance(data, dict) and "value_cm" in data:
            conf = data["confidence"]
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            table.add_row(
                field,
                str(data["value_cm"]),
                f"{bar} {conf:.0%}",
                data["source"],
            )

    console.print(table)


@app.command()
def benchmark(
    dataset_dir: Path = typer.Option(
        Path("data/benchmarks"),
        "--data",
        help="Benchmark dataset directory",
    ),
) -> None:
    """Run accuracy benchmark against ground truth data."""
    import subprocess

    console.print("[bold]Running accuracy benchmark...[/bold]")
    result = subprocess.run(
        ["pytest", "tests/benchmark/", "-v", "--tb=short"],
        capture_output=False,
    )
    raise typer.Exit(result.returncode)


@app.command()
def retention(
    days: int = typer.Option(
        30, "--days", "-d", help="Delete avatars older than N days"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count without deleting"),
) -> None:
    """Enforce the avatar retention policy.

    Schedule this via cron / systemd-timer / Celery beat — body scans are
    sensitive PII and must not linger past the retention window.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy.orm import Session

    from fitscan.infrastructure.result_store import (
        AvatarRecord,
        AvatarStore,
        engine,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if dry_run:
        with Session(engine) as session:
            count = (
                session.query(AvatarRecord)
                .filter(AvatarRecord.created_at < cutoff)
                .count()
            )
        console.print(
            f"[yellow]dry-run:[/yellow] would delete [bold]{count}[/bold] "
            f"avatars older than {days}d"
        )
        return

    removed = AvatarStore().delete_older_than(days)
    console.print(
        f"[green]retention:[/green] deleted [bold]{removed}[/bold] "
        f"avatars older than {days}d"
    )


if __name__ == "__main__":
    app()
