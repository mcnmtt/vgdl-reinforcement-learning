"""
Genera grafici e un report HTML dai risultati di evaluate_model_testset.py.

Esempi:
  python evaluation/plot_evaluation_results.py

  python evaluation/plot_evaluation_results.py \
      --runs gemma4-zero-shot gemma4-supervised gemma4-rl

  python evaluation/plot_evaluation_results.py \
      --runs gemma4-zero-shot gemma4-supervised \
      --output-dir evaluation/plots/gemma4-comparison
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "plots"

RATE_METRICS = {
    "executability_rate": "Executability",
    "basic_game_rate": "Starts with BasicGame",
    "complete_structure_rate": "Complete structure",
}

SIMILARITY_METRICS = {
    "mean_sprite_similarity": "Sprites",
    "mean_interaction_similarity": "Interactions",
    "mean_termination_similarity": "Terminations",
    "mean_structural_similarity": "Overall",
}

DETAIL_METRICS = {
    "sprite_similarity": "Sprites",
    "interaction_similarity": "Interactions",
    "termination_similarity": "Terminations",
    "final_score": "Overall",
}

COLORS = ["#3366CC", "#DC3912", "#FF9900", "#109618", "#990099"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizza e confronta i risultati delle valutazioni VGDL."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help=(
            "Nomi delle cartelle dentro evaluation/results. "
            "Se omesso, usa tutti i run disponibili."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra anche i grafici interattivamente.",
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, list[dict]]:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    details_path = run_dir / "details.jsonl"

    missing = [
        str(path.name)
        for path in (summary_path, metrics_path, details_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' incompleto. File mancanti: {missing}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    details = [
        json.loads(line)
        for line in details_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics["run_name"] = summary["run_name"]
    return summary, metrics, details


def save_figure(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, decimals: int = 2) -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=f"%.{decimals}f",
            fontsize=8,
            padding=2,
        )


def plot_rates(summaries: list[dict], output_dir: Path, show: bool) -> Path:
    labels = [summary["run_name"] for summary in summaries]
    x = np.arange(len(labels))
    width = 0.8 / len(RATE_METRICS)

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2.4), 5.5))
    for offset, (metric, title) in enumerate(RATE_METRICS.items()):
        values = [summary.get(metric, 0.0) for summary in summaries]
        positions = x - 0.4 + width / 2 + offset * width
        ax.bar(
            positions,
            values,
            width,
            label=title,
            color=COLORS[offset],
        )

    ax.set_title("VGDL validity and format compliance")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    add_bar_labels(ax)

    path = output_dir / "01_validity_and_structure.png"
    save_figure(fig, path, show)
    return path


def plot_similarity(
    summaries: list[dict],
    output_dir: Path,
    show: bool,
) -> Path:
    labels = [summary["run_name"] for summary in summaries]
    x = np.arange(len(labels))
    width = 0.8 / len(SIMILARITY_METRICS)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.7), 5.5))
    for offset, (metric, title) in enumerate(
        SIMILARITY_METRICS.items()
    ):
        values = [summary.get(metric, 0.0) for summary in summaries]
        positions = x - 0.4 + width / 2 + offset * width
        ax.bar(
            positions,
            values,
            width,
            label=title,
            color=COLORS[offset],
        )

    ax.set_title("Mean structural similarity to reference VGDL")
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    add_bar_labels(ax)

    path = output_dir / "02_similarity_comparison.png"
    save_figure(fig, path, show)
    return path


def plot_per_example(
    metrics_by_run: dict[str, pd.DataFrame],
    output_dir: Path,
    show: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, (run_name, metrics) in enumerate(metrics_by_run.items()):
        ax.plot(
            metrics["index"],
            metrics["final_score"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=run_name,
            color=COLORS[index % len(COLORS)],
        )

    ax.set_title("Structural similarity for each test example")
    ax.set_xlabel("Test-set example index")
    ax.set_ylabel("Overall similarity")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(
        sorted(
            {
                int(value)
                for metrics in metrics_by_run.values()
                for value in metrics["index"]
            }
        )
    )
    ax.grid(alpha=0.25)
    ax.legend()

    path = output_dir / "03_similarity_per_example.png"
    save_figure(fig, path, show)
    return path


def plot_similarity_heatmap(
    metrics_by_run: dict[str, pd.DataFrame],
    output_dir: Path,
    show: bool,
) -> Path:
    run_names = list(metrics_by_run)
    all_indices = sorted(
        {
            int(value)
            for metrics in metrics_by_run.values()
            for value in metrics["index"]
        }
    )
    matrix = np.full((len(run_names), len(all_indices)), np.nan)
    index_position = {
        example_index: position
        for position, example_index in enumerate(all_indices)
    }

    for row, run_name in enumerate(run_names):
        for _, record in metrics_by_run[run_name].iterrows():
            column = index_position[int(record["index"])]
            matrix[row, column] = record["final_score"]

    fig, ax = plt.subplots(
        figsize=(max(11, len(all_indices) * 0.55), max(3.5, len(run_names)))
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Overall similarity heatmap")
    ax.set_xlabel("Test-set example index")
    ax.set_yticks(range(len(run_names)), run_names)
    ax.set_xticks(range(len(all_indices)), all_indices)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isnan(value):
                color = "white" if value > 0.55 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    fig.colorbar(image, ax=ax, label="Overall similarity")
    path = output_dir / "04_similarity_heatmap.png"
    save_figure(fig, path, show)
    return path


def plot_generation_performance(
    metrics_by_run: dict[str, pd.DataFrame],
    output_dir: Path,
    show: bool,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    time_data = [
        metrics["generation_seconds"].dropna().to_numpy()
        for metrics in metrics_by_run.values()
    ]
    axes[0].boxplot(time_data, tick_labels=list(metrics_by_run))
    axes[0].set_title("Generation time distribution")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)

    for index, (run_name, metrics) in enumerate(metrics_by_run.items()):
        axes[1].scatter(
            metrics["generated_tokens"],
            metrics["generation_seconds"],
            label=run_name,
            alpha=0.75,
            color=COLORS[index % len(COLORS)],
        )
    axes[1].set_title("Output length versus generation time")
    axes[1].set_xlabel("Generated tokens")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    path = output_dir / "05_generation_performance.png"
    save_figure(fig, path, show)
    return path


def create_comparison_table(summaries: list[dict], path: Path) -> pd.DataFrame:
    columns = [
        "run_name",
        "backend",
        "model",
        "examples",
        "executability_rate",
        "complete_structure_rate",
        "mean_sprite_similarity",
        "mean_interaction_similarity",
        "mean_termination_similarity",
        "mean_structural_similarity",
        "mean_generation_seconds",
    ]
    table = pd.DataFrame(summaries)
    for column in columns:
        if column not in table:
            table[column] = None
    table = table[columns]
    table.to_csv(path, index=False)
    return table


def truncate(text: str, maximum: int = 600) -> str:
    text = text.strip()
    if len(text) <= maximum:
        return text
    return text[:maximum] + "\n[...]"


def build_details_html(details_by_run: dict[str, list[dict]]) -> str:
    sections = []
    for run_name, records in details_by_run.items():
        rows = []
        for record in sorted(records, key=lambda item: item["index"]):
            errors = "\n".join(record.get("validation_errors", [])) or "None"
            generated = truncate(record.get("generated_vgdl", ""))
            reference = truncate(record.get("reference_vgdl", ""))
            rows.append(
                f"""
                <tr>
                  <td>{record["index"]}</td>
                  <td>{record.get("valid", False)}</td>
                  <td>{record.get("all_sections_present", False)}</td>
                  <td>{record.get("sprite_similarity", 0):.3f}</td>
                  <td>{record.get("interaction_similarity", 0):.3f}</td>
                  <td>{record.get("termination_similarity", 0):.3f}</td>
                  <td><strong>{record.get("final_score", 0):.3f}</strong></td>
                  <td>{record.get("generation_seconds", 0):.2f}s</td>
                  <td>
                    <details><summary>Errors</summary>
                      <pre>{html.escape(errors)}</pre>
                    </details>
                    <details><summary>Generated VGDL</summary>
                      <pre>{html.escape(generated)}</pre>
                    </details>
                    <details><summary>Reference VGDL</summary>
                      <pre>{html.escape(reference)}</pre>
                    </details>
                  </td>
                </tr>
                """
            )

        sections.append(
            f"""
            <h2>{html.escape(run_name)}</h2>
            <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Index</th><th>Valid</th><th>Structure</th>
                  <th>Sprites</th><th>Interactions</th>
                  <th>Terminations</th><th>Overall</th>
                  <th>Time</th><th>Details</th>
                </tr>
              </thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
            </div>
            """
        )
    return "\n".join(sections)


def write_html_report(
    summaries: list[dict],
    comparison_table: pd.DataFrame,
    details_by_run: dict[str, list[dict]],
    plot_paths: list[Path],
    output_path: Path,
) -> None:
    summary_rows = []
    for _, row in comparison_table.iterrows():
        summary_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in row
            )
            + "</tr>"
        )

    headers = "".join(
        f"<th>{html.escape(column)}</th>"
        for column in comparison_table.columns
    )
    plots = "\n".join(
        f'<section><img src="{path.name}" alt="{path.stem}"></section>'
        for path in plot_paths
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VGDL Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 30px; color: #222; }}
    h1, h2 {{ color: #17365d; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 28px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 7px; text-align: left; }}
    th {{ background: #eaf1f8; position: sticky; top: 0; }}
    pre {{ white-space: pre-wrap; max-width: 900px; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 35px; }}
    details {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>VGDL Model Evaluation</h1>
  <p>Runs compared: {html.escape(", ".join(s["run_name"] for s in summaries))}</p>
  <h2>Summary</h2>
  <div class="table-wrap">
    <table><thead><tr>{headers}</tr></thead>
    <tbody>{''.join(summary_rows)}</tbody></table>
  </div>
  <h2>Plots</h2>
  {plots}
  <h1>Per-example details</h1>
  {build_details_html(details_by_run)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        run_dirs = [results_dir / run_name for run_name in args.runs]
    else:
        run_dirs = sorted(
            path
            for path in results_dir.iterdir()
            if path.is_dir() and (path / "summary.json").exists()
        )

    if not run_dirs:
        raise SystemExit(f"Nessun run trovato in {results_dir}")

    summaries = []
    metrics_by_run = {}
    details_by_run = {}
    for run_dir in run_dirs:
        summary, metrics, details = load_run(run_dir)
        run_name = summary["run_name"]
        summaries.append(summary)
        metrics_by_run[run_name] = metrics
        details_by_run[run_name] = details

    plot_paths = [
        plot_rates(summaries, output_dir, args.show),
        plot_similarity(summaries, output_dir, args.show),
        plot_per_example(metrics_by_run, output_dir, args.show),
        plot_similarity_heatmap(metrics_by_run, output_dir, args.show),
        plot_generation_performance(
            metrics_by_run,
            output_dir,
            args.show,
        ),
    ]

    comparison_path = output_dir / "comparison.csv"
    comparison_table = create_comparison_table(
        summaries,
        comparison_path,
    )
    report_path = output_dir / "report.html"
    write_html_report(
        summaries,
        comparison_table,
        details_by_run,
        plot_paths,
        report_path,
    )

    print(f"Run analizzati: {', '.join(metrics_by_run)}")
    print(f"Grafici salvati in: {output_dir}")
    print(f"Tabella comparativa: {comparison_path}")
    print(f"Report HTML: {report_path}")


if __name__ == "__main__":
    main()
