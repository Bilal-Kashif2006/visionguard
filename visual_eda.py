def generate_visual_eda(df: pd.DataFrame, samples_per_class: int = 4):
  """Generates sample image grids and per-class histograms for width, height, and file size."""
  sns.set_theme(style="whitegrid")
  classes = df["class"].unique()

  # 1. Sample Grid per Class
  fig, axes = plt.subplots(
      len(classes),
      samples_per_class,
      figsize=(samples_per_class * 3, len(classes) * 3),
  )
  for row_idx, cls in enumerate(classes):
    cls_df = df[df["class"] == cls]
    sampled_paths = cls_df.sample(
        n=min(samples_per_class, len(cls_df)), random_state=42
    )["filepath"].tolist()

    for col_idx in range(samples_per_class):
      ax = (
          axes[row_idx, col_idx]
          if len(classes) > 1
          else axes[col_idx]
      )
      if col_idx < len(sampled_paths):
        img = Image.open(sampled_paths[col_idx])
        ax.imshow(img)
        ax.set_title(f"{cls}\n{img.width}x{img.height}", fontsize=9)
      ax.axis("off")

  plt.suptitle(
      "Figure 1: Grid of Representative Sample Images per Class",
      fontsize=14,
      y=1.02,
  )
  plt.tight_layout()
  plt.savefig("fig1_sample_grid.png", bbox_inches="tight")
  plt.close()

  # 2. Width Histogram per Class
  plt.figure(figsize=(8, 5))
  sns.histplot(
      data=df,
      x="width",
      hue="class",
      bins=30,
      kde=True,
      element="step",
      stat="density",
      common_norm=False,
  )
  plt.title("Figure 2: Distribution of Image Width by Class")
  plt.xlabel("Width (pixels)")
  plt.ylabel("Density")
  plt.tight_layout()
  plt.savefig("fig2_width_histogram.png")
  plt.close()

  # 3. Height Histogram per Class
  plt.figure(figsize=(8, 5))
  sns.histplot(
      data=df,
      x="height",
      hue="class",
      bins=30,
      kde=True,
      element="step",
      stat="density",
      common_norm=False,
  )
  plt.title("Figure 3: Distribution of Image Height by Class")
  plt.xlabel("Height (pixels)")
  plt.ylabel("Density")
  plt.tight_layout()
  plt.savefig("fig3_height_histogram.png")
  plt.close()

  # 4. File Size Histogram per Class
  plt.figure(figsize=(8, 5))
  sns.histplot(
      data=df,
      x="file_size_kb",
      hue="class",
      bins=30,
      kde=True,
      element="step",
      stat="density",
      common_norm=False,
  )
  plt.title("Figure 4: Distribution of Image File Size by Class")
  plt.xlabel("File Size (KB)")
  plt.ylabel("Density")
  plt.tight_layout()
  plt.savefig("fig4_filesize_histogram.png")
  plt.close()


# Usage:
# generate_visual_eda(df_audit)