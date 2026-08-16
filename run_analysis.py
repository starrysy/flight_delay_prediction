"""Run the repository's reproducible model comparison and evaluation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flight_delay.modeling import run_analysis  # noqa: E402


if __name__ == "__main__":
    summary = run_analysis(
        Path("data/raw/flights_january_2019.csv"),
        Path("results"),
    )
    print(f"Selected model: {summary['selected_model']}")
    for metric, value in summary["test_metrics"].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
