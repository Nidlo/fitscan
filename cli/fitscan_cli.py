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

app = typer.Typer(
    name="fitscan", help="AI-powered body measurement from photos."
)
console = Console()


@app.command()
def measure(
    image: Path = typer.Argument(..., help="Path to photo"),
    height: float | None = typer.Option(
        None, "--height", "-h", help="Height in cm"
    ),
    reference: float | None = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference object long side in cm",
    ),
    no_smpl: bool = typer.Option(
        False, "--no-smpl", help="Disable SMPL fitting"
    ),
    no_claude: bool = typer.Option(
        False, "--no-claude", help="Disable Claude validation"
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output raw JSON"
    ),
    gender: str = typer.Option(
        "ladies", "--gender", "-g", help="Gender profile"
    ),
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


if __name__ == "__main__":
    app()
