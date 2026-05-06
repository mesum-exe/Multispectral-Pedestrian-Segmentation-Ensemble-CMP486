all_ir_train = sorted(llvip_infrared_train_dir.glob("*.jpg"))
all_vis_train = sorted(llvip_visible_train_dir.glob("*.jpg"))
print(f"Infrared training images: {len(all_ir_train)}")
print(f"Visible training images: {len(all_vis_train)}")

ir_stems = {p.stem for p in all_ir_train}
vis_stems = {p.stem for p in all_vis_train}
ir_only = ir_stems - vis_stems
vis_only = vis_stems - ir_stems

if not ir_only and not vis_only:
    print("Alignment check passed. Every IR image has a matching visible pair.")
else:
    print(f"Alignment check failed. IR-only: {len(ir_only)} | visible-only: {len(vis_only)}")


sample_idx = random.sample(range(len(all_ir_train)), 3)
print(f"Selected {len(sample_idx)} image pairs for the alignment plot.")


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("LLVIP: Infrared | Visible | Fused Overlay", fontsize=14, fontweight="bold")

for row, idx in enumerate(sample_idx):
    ir_gray = cv2.imread(str(all_ir_train[idx]), cv2.IMREAD_GRAYSCALE)
    vis_bgr = cv2.imread(str(all_vis_train[idx]))
    if ir_gray is None or vis_bgr is None:
        raise FileNotFoundError(f"Could not read pair: {all_ir_train[idx].name}")

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    heatmap = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    vis_resized = cv2.resize(vis_rgb, (ir_gray.shape[1], ir_gray.shape[0]))
    fused_overlay = cv2.addWeighted(vis_resized, 0.6, heatmap, 0.4, 0)

    axes[row, 0].imshow(ir_gray, cmap="gray")
    axes[row, 0].set_title("Infrared")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(vis_rgb)
    axes[row, 1].set_title("Visible")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(fused_overlay)
    axes[row, 2].set_title("Fused overlay")
    axes[row, 2].axis("off")

plt.tight_layout()
alignment_plot_path = results_dir / "section4_alignment_check.png"
plt.savefig(alignment_plot_path, dpi=150, bbox_inches="tight")
plt.show()

# Save our plot in a pre-defined path
print(f"Saved alignment plot: {alignment_plot_path}")
