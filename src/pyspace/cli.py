"""Command-line interface for reproducible PySpace workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core.census import CensusResult, census_coordinates, census_image, export_multi_image_census_dataframe
from .core.pattern_learning import learn_pattern_result
from .core.pattern_models import PatternResult
from .core.r_measure_cismi import measure_cisMI
from .io.table_loader import read_table
from .serialization import load_result, save_result

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".parquet"}
_PICKLE_SUFFIXES = {".pkl", ".pickle"}


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyspace", description="Spatial Patterning Analysis of Cellular Ensembles")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--verbose", "-v", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    census = commands.add_parser("census", help="Collect neighborhoods from an image or coordinate table")
    census.add_argument("input")
    census.add_argument("--radii", "-r", required=True, help="Comma-separated positive radii")
    census.add_argument("--variables", help="Comma-separated table variables")
    census.add_argument("--sample-size", required=True, help="One count or a comma-separated count per radius")
    census.add_argument("--format", choices=["bundle", "csv", "json"], default="bundle")
    census.add_argument("--output", "-o", required=True)

    analyze = commands.add_parser("analyze", help="Run pattern learning or cisMI")
    analyze.add_argument("input")
    analyze.add_argument("--method", "-m", choices=["som", "cisMI"], default="som")
    analyze.add_argument("--variables", help="Comma-separated variables")
    analyze.add_argument("--radii", help="Comma-separated cisMI radii")
    analyze.add_argument("--depth", type=int, default=3)
    analyze.add_argument("--bootstraps", type=int, default=100)
    analyze.add_argument("--max-bins", type=int, default=100)
    analyze.add_argument("--cores", type=int)
    analyze.add_argument("--random-state", type=int)
    analyze.add_argument("--allow-permutation-fallback", action="store_true")
    analyze.add_argument("--all-vars", help="Comma-separated variables required in every max-depth ensemble")
    analyze.add_argument("--alo-vars", help="Comma-separated variables, at least one of which is required")
    analyze.add_argument("--not-vars", help="Comma-separated excluded variables")
    analyze.add_argument("--patch-list", help="Safe result bundle containing a patch list")
    analyze.add_argument("--iterations", type=int, default=50)
    analyze.add_argument("--format", choices=["bundle", "csv", "json"], default="bundle")
    analyze.add_argument("--output", "-o", required=True)

    plot = commands.add_parser("plot", help="Render a saved result")
    plot.add_argument("input")
    plot.add_argument("--type", "-t", choices=["covariation", "enrichment", "som"], default="covariation")
    plot.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    plot.add_argument("--dpi", type=int, default=300)
    plot.add_argument("--width", type=float, default=10.0)
    plot.add_argument("--height", type=float, default=6.0)
    plot.add_argument("--output", "-o", required=True)

    convert = commands.add_parser("convert", help="Convert safe table and result formats")
    convert.add_argument("input")
    convert.add_argument("--format", "-f", choices=["bundle", "csv", "json", "npz"], required=True)
    convert.add_argument("--output", "-o", required=True)
    return parser


def parse_comma_separated(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def parse_radii(value: str) -> list[float]:
    try:
        radii = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid radii {value!r}: {exc}") from exc
    if not radii or any(radius <= 0 or not np.isfinite(radius) for radius in radii):
        raise ValueError("radii must contain finite positive numbers")
    return radii


def _parse_sample_size(value: str, radius_count: int) -> int | list[int]:
    try:
        counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid sample size {value!r}: {exc}") from exc
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("sample size must contain positive integers")
    if len(counts) not in {1, radius_count}:
        raise ValueError("sample size must contain one count or one count per radius")
    return counts[0] if len(counts) == 1 else counts


def handle_census_command(args: argparse.Namespace) -> None:
    source = _existing_path(args.input)
    radii = parse_radii(args.radii)
    variables = parse_comma_separated(args.variables) or None
    sample_size = _parse_sample_size(args.sample_size, len(radii))
    suffix = source.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        if variables is not None:
            raise ValueError("--variables is only supported for coordinate-table census")
        result = census_image(
            source,
            radii=radii,
            sample_size=sample_size,
        )
    elif suffix in _TABLE_SUFFIXES:
        result = census_coordinates(
            read_table(source),
            radii=radii,
            variables=variables,
            sample_size=sample_size,
        )
    else:
        raise ValueError(f"Unsupported census input format: {source.suffix or '<none>'}")
    save_results(result, args.output, args.format, args.verbose)


def _extract_census(value: Any) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None]:
    if isinstance(value, pd.DataFrame):
        return value, None
    if isinstance(value, CensusResult):
        if value.census is not None:
            return value.census, value.patch_list
        return export_multi_image_census_dataframe(value.neighborhoods, value.variables), value.patch_list
    if isinstance(value, dict):
        census = value.get("census")
        if isinstance(census, pd.DataFrame):
            patches = value.get("patch_list")
            return census, patches if isinstance(patches, dict) else None
    raise ValueError("Input does not contain a census table")


def handle_analyze_command(args: argparse.Namespace) -> None:
    data = load_data(args.input, args.verbose)
    variables = parse_comma_separated(args.variables)
    result: Any
    if args.method == "cisMI":
        census, stored_patch_list = _extract_census(data)
        patch_list = stored_patch_list
        if args.patch_list:
            patch_payload = load_data(args.patch_list, args.verbose)
            patch_list = patch_payload.get("patch_list", patch_payload) if isinstance(patch_payload, dict) else None
            if not isinstance(patch_list, dict):
                raise ValueError("--patch-list must contain a patch-list mapping")
        result = measure_cisMI(
            census,
            patch_list=patch_list,
            depth=args.depth,
            radii=parse_radii(args.radii) if args.radii else None,
            bootstraps=args.bootstraps,
            all=parse_comma_separated(args.all_vars) or None,
            alo=parse_comma_separated(args.alo_vars) or None,
            not_=parse_comma_separated(args.not_vars) or None,
            max_bins=args.max_bins,
            cores=args.cores,
            allow_permutation_fallback=args.allow_permutation_fallback,
            random_state=args.random_state,
        )
    else:
        if not variables:
            raise ValueError("--variables is required for pattern analysis")
        census, _ = _extract_census(data)
        if "Radius" not in census:
            raise ValueError("Pattern analysis requires a Radius column")
        radius = parse_radii(args.radii)[0] if args.radii else float(census["Radius"].iloc[0])
        result = learn_pattern_result(
            census,
            variables,
            radius,
            {"variables": ["#000000"]},
            som_reps=args.iterations,
            random_state=args.random_state,
        )
    save_results(result, args.output, args.format, args.verbose)


def _pattern_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, PatternResult):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise ValueError("Plot input is not a pattern result")


def handle_plot_command(args: argparse.Namespace) -> None:
    result = _pattern_mapping(load_data(args.input, args.verbose))
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(args.width, args.height))
    if args.type == "enrichment":
        axis.plot(np.asarray(result["enrichment_scores"], dtype=float))
        axis.set(xlabel="SOM node", ylabel="Enrichment score")
    elif args.type == "som":
        history = result["som_result"]["training_history"]
        axis.plot(history["quantization_error"])
        axis.set(xlabel="Iteration", ylabel="Quantization error")
    else:
        covariation = result.get("covariation_data")
        if not isinstance(covariation, pd.DataFrame):
            raise ValueError("Pattern result does not contain covariation data")
        for variable, frame in covariation.groupby("variable", sort=False):
            axis.plot(frame["position"], frame["mean_abundance"], label=str(variable))
        axis.legend()
        axis.set(xlabel="SOM position", ylabel="Mean abundance")
    figure.tight_layout()
    figure.savefig(destination, dpi=args.dpi, format=args.format)
    plt.close(figure)


def handle_convert_command(args: argparse.Namespace) -> None:
    save_results(load_data(args.input, args.verbose), args.output, args.format, args.verbose)


def _existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def load_data(file_path: str | Path, verbose: bool = False) -> Any:
    """Load an explicitly supported safe format; pickle is always rejected."""
    path = _existing_path(file_path)
    suffix = path.suffix.lower()
    if suffix in _PICKLE_SUFFIXES:
        raise ValueError("Unsafe pickle input is not supported; use a PySpace result bundle")
    if path.is_dir() or suffix == ".pyspace":
        result = load_result(path)
    elif suffix in _TABLE_SUFFIXES:
        result = read_table(path)
    elif suffix == ".json":
        result = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            result = {name: np.asarray(arrays[name]) for name in arrays.files}
    else:
        raise ValueError(f"Unsupported input format: {path.suffix or '<none>'}")
    if verbose:
        print(f"Loaded: {path}")
    return result


def _csv_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, CensusResult):
        if data.census is not None:
            return data.census
        return export_multi_image_census_dataframe(data.neighborhoods, data.variables)
    if isinstance(data, dict) and len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, pd.DataFrame):
            return only_value
    raise TypeError("CSV output requires a DataFrame or one tabular result")


def _json_value(data: Any) -> Any:
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if is_dataclass(data):
        return asdict(cast(Any, data))
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, Path):
        return str(data)
    raise TypeError(f"Value of type {type(data).__name__} is not JSON serializable")


def save_results(data: Any, output_path: str | Path, format_type: str, verbose: bool = False) -> None:
    """Write one supported format without overwriting an existing output."""
    destination = Path(output_path).expanduser()
    if format_type in {"pkl", "pickle"}:
        raise ValueError("Unsafe pickle output is not supported; use a PySpace result bundle")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if format_type == "bundle":
        save_result(data, destination)
    elif format_type == "csv":
        _csv_frame(data).to_csv(destination, index=False)
    elif format_type == "json":
        destination.write_text(
            json.dumps(data, default=_json_value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif format_type == "npz":
        if isinstance(data, np.ndarray):
            arrays = {"array": data}
        elif isinstance(data, dict) and all(isinstance(value, np.ndarray) for value in data.values()):
            arrays = data
        else:
            raise TypeError("NPZ output requires an array or a mapping of arrays")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise TypeError("NPZ output does not support object-dtype arrays")
        cast(Any, np.savez_compressed)(destination, **arrays)
    else:
        raise ValueError(f"Unsupported output format: {format_type}")
    if verbose:
        print(f"Saved: {destination}")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a stable process exit code."""
    args = setup_argument_parser().parse_args(argv)
    handlers = {
        "census": handle_census_command,
        "analyze": handle_analyze_command,
        "plot": handle_plot_command,
        "convert": handle_convert_command,
    }
    try:
        handlers[args.command](args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - genuine process boundary with stable error reporting
        print(f"Operation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
