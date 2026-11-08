"""Dataset audit and visual EDA for the AI-generated-vs-real image dataset."""

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError


AUDIT_COLUMNS = [
    "filepath",
    "class",
    "file_size_bytes",
    "file_size_kb",
    "width",
    "height",
    "colour_mode",
    "format",
    "sha256",
]


def _sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with filepath.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract image metadata and return (readable_images, corrupt_files)."""
    records: list[dict] = []
    corrupt_files: list[dict] = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    for filepath in sorted(dataset_path.rglob("*")):
        if not filepath.is_file():
            continue

        try:
            file_hash = _sha256(filepath)
            with Image.open(filepath) as image:
                # load() forces PIL to read the pixel data, catching truncated files.
                image.load()
                width, height = image.size
                colour_mode = image.mode
                image_format = image.format or filepath.suffix[1:].upper() or "UNKNOWN"
        except (OSError, UnidentifiedImageError, ValueError) as error:
            corrupt_files.append(
                {"filepath": str(filepath), "error": f"{type(error).__name__}: {error}"}
            )
            continue

        file_size_bytes = filepath.stat().st_size
        records.append(
            {
                "filepath": str(filepath),
                "class": filepath.parent.name,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": file_size_bytes / 1024,
                "width": width,
                "height": height,
                "colour_mode": colour_mode,
                "format": image_format,
                "sha256": file_hash,
            }
        )

    return (
        pd.DataFrame(records, columns=AUDIT_COLUMNS),
        pd.DataFrame(corrupt_files, columns=["filepath", "error"]),
    )


def generate_audit_report(
    df: pd.DataFrame, df_corrupt: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Compute class counts, per-class min/median/max, and exact duplicates."""
    if df.empty:
        df_duplicates = pd.DataFrame(columns=AUDIT_COLUMNS)
        class_counts = pd.DataFrame(columns=["Class", "Image Count"])
        stats = pd.DataFrame(
            columns=[
                "Class",
                "Width Min",
                "Width Med",
                "Width Max",
                "Height Min",
                "Height Med",
                "Height Max",
                "Size Min (KB)",
                "Size Med (KB)",
                "Size Max (KB)",
            ]
        )
        return class_counts, stats, df_duplicates, len(df_corrupt)

    duplicate_mask = df.duplicated(subset=["sha256"], keep=False)
    df_duplicates = df.loc[duplicate_mask].sort_values(
        ["sha256", "filepath"]
    )
    duplicate_files = int(df.duplicated(subset=["sha256"], keep="first").sum())

    class_counts = (
        df["class"].value_counts().rename_axis("Class").reset_index(name="Image Count")
    )
    stats = []
    for class_name, group in df.groupby("class", sort=True):
        stats.append(
            {
                "Class": class_name,
                "Width Min": int(group["width"].min()),
                "Width Med": float(group["width"].median()),
                "Width Max": int(group["width"].max()),
                "Height Min": int(group["height"].min()),
                "Height Med": float(group["height"].median()),
                "Height Max": int(group["height"].max()),
                "Size Min (KB)": round(group["file_size_kb"].min(), 2),
                "Size Med (KB)": round(group["file_size_kb"].median(), 2),
                "Size Max (KB)": round(group["file_size_kb"].max(), 2),
            }
        )

    return class_counts, pd.DataFrame(stats), df_duplicates, duplicate_files


def create_eda_figures(
    df: pd.DataFrame, output_dir: Path, samples_per_class: int = 5
) -> None:
    """Create the requested sample grid and per-class metadata histograms."""
    sns.set_theme(style="whitegrid")
    classes = sorted(df["class"].unique())

    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(4 * samples_per_class, 4 * max(len(classes), 1)),
        squeeze=False,
    )
    for row, class_name in enumerate(classes):
        sample = df[df["class"] == class_name].head(samples_per_class)
        for col in range(samples_per_class):
            axis = axes[row][col]
            axis.axis("off")
            if col >= len(sample):
                continue
            filepath = sample.iloc[col]["filepath"]
            try:
                with Image.open(filepath) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(class_name if col == 0 else "")
            except (OSError, UnidentifiedImageError) as error:
                axis.set_title(f"Unreadable: {type(error).__name__}")
    fig.suptitle("Sample images by class", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "Caption: A random-looking, deterministic sample of images from each class.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_dir / "sample_images_by_class.png", dpi=150)
    plt.close(fig)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, column, label in zip(
        axes,
        ["width", "height", "file_size_kb"],
        ["Width (pixels)", "Height (pixels)", "File size (KB)"],
    ):
        sns.histplot(
            data=df,
            x=column,
            hue="class",
            element="step",
            stat="count",
            common_norm=False,
            ax=axis,
        )
        axis.set_title(f"{label} distribution by class")
        axis.set_xlabel(label)
        axis.set_ylabel("Image count")
    figure.suptitle("Image metadata distributions", fontsize=16)
    figure.text(
        0.5,
        0.01,
        "Caption: Histograms compare width, height, and file-size distributions for the two classes.",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    figure.savefig(output_dir / "metadata_histograms_by_class.png", dpi=150)
    plt.close(figure)


def write_interpretations(
    df: pd.DataFrame,
    class_counts: pd.DataFrame,
    stats: pd.DataFrame,
    duplicate_files: int,
    corrupt_count: int,
    output_dir: Path,
) -> None:
    """Write concise, data-derived figure interpretations for the report."""
    lines = [
        "Figure interpretations",
        "",
        "Sample image grid: The grid shows representative examples from every class. "
        "It is useful for checking that folder names match the visible content and for spotting "
        "obvious outliers before modelling.",
        "",
        "Metadata histograms: The distributions show how image dimensions and storage sizes vary "
        "within and between classes. Differences in these distributions can indicate collection "
        "or preprocessing bias, so resizing and normalization should be applied consistently.",
        "",
        "Audit interpretation: "
        f"{len(df)} readable images were found across {len(class_counts)} classes; "
        f"{corrupt_count} files were unreadable and {duplicate_files} files were exact duplicates "
        "(duplicate copies beyond the first occurrence).",
        "",
        "Per-class statistics:",
        stats.to_string(index=False),
    ]
    (output_dir / "interpretations.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=project_dir / "archive",
        type=Path,
        help="Dataset root (default: ./archive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "audit_output",
        help="Directory for CSV reports and figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_audit, df_corrupt = audit_dataset(args.dataset_dir)
    class_counts, df_stats, df_duplicates, duplicate_files = generate_audit_report(
        df_audit, df_corrupt
    )

    df_audit.to_csv(args.output_dir / "image_audit.csv", index=False)
    df_corrupt.to_csv(args.output_dir / "corrupt_files.csv", index=False)
    df_duplicates.to_csv(args.output_dir / "exact_duplicates.csv", index=False)
    class_counts.to_csv(args.output_dir / "class_counts.csv", index=False)
    df_stats.to_csv(args.output_dir / "class_statistics.csv", index=False)

    if not df_audit.empty:
        create_eda_figures(df_audit, args.output_dir)
    write_interpretations(
        df_audit,
        class_counts,
        df_stats,
        duplicate_files,
        len(df_corrupt),
        args.output_dir,
    )

    print("=== DATASET AUDIT SUMMARY ===")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Total processed images: {len(df_audit)}")
    print(f"Total corrupt / unreadable files: {len(df_corrupt)}")
    print(f"Exact duplicate files beyond first occurrence: {duplicate_files}")
    print(f"Exact duplicate rows: {len(df_duplicates)}")
    print("\n--- Class counts ---")
    print(class_counts.to_string(index=False))
    print("\n--- Width, height and file-size statistics per class ---")
    print(df_stats.to_string(index=False))
    print(f"\nReports and figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
