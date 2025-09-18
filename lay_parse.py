import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import layoutparser as lp
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

def load_layout_model():
    model = lp.Detectron2LayoutModel(
        config_path="lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        score_thresh= 0.5
        )
    return model

doctr_predictor = ocr_predictor(pretrained=True)

# --- Detect layout ---
def detect_layout(image_path: str, model):
    image = Image.open(image_path).convert("RGB")
    layout = model.detect(np.array(image))
    # filter by score threshold
    filtered = lp.Layout([
        b for b in layout if b.score is not None and b.score >= 0.5
    ])
    return image, filtered

# --- Cluster blocks into columns ---
def cluster_into_columns(layout: lp.Layout, n_columns=None):
    if len(layout) == 0:
        return []

    centers = np.array([(b.block.x_1 + b.block.x_2) / 2.0 for b in layout])
    idx_sorted = np.argsort(centers)
    sorted_boxes = [layout[i] for i in idx_sorted]

    if n_columns is None:
        diffs = np.diff(np.sort(centers))
        median_gap = np.median(diffs) if len(diffs) > 0 else 0
        threshold = max(1.5 * median_gap, median_gap + 1e-6)
        split_indices = np.where(diffs > threshold)[0] + 1
        groups = np.split(sorted_boxes, split_indices)
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_columns, random_state=0).fit(centers.reshape(-1, 1))
        groups = []
        for k in range(n_columns):
            members = [layout[i] for i, label in enumerate(kmeans.labels_) if label == k]
            groups.append(members)

    columns = []
    for group in groups:
        group_sorted = sorted(group, key=lambda b: b.block.y_1)
        columns.append(lp.Layout(group_sorted))
    return columns

# --- OCR with doctr ---
def ocr_layout_blocks(image_pil: Image.Image, layout: lp.Layout):
    img_cv = np.array(image_pil)
    results = []
    for block in layout:
        x1, y1, x2, y2 = map(int, (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
        crop = img_cv[y1:y2, x1:x2]
        if crop.size == 0:
            results.append((block, ""))
            continue
        doc = DocumentFile.from_images([crop])
        ocr_result = doctr_predictor(doc)
        text = " ".join(
            [w.value for page in ocr_result.pages for b in page.blocks for l in b.lines for w in l.words]
        )
        results.append((block, text.strip()))
    return results

# --- Visualization ---
def visualize_layout(image_pil, layout, ocr_texts=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(image_pil)
    for i, block in enumerate(layout):
        x1, y1, x2, y2 = block.coordinates
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)
        label = f"{block.type}:{block.score:.2f}" if block.score is not None else block.type
        ax.text(x1, y1 - 6, label, fontsize=8, color="red",
                bbox=dict(facecolor="yellow", alpha=0.3, pad=1, edgecolor='none'))
        if ocr_texts:
            _, text = ocr_texts[i]
            if text:
                snippet = text.replace("\n", " ")[:120]
                ax.text(x1, y2 + 6, snippet, fontsize=7, color="black",
                        bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor='none'))
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# --- Save JSON ---
def save_as_json(results, output_json: str):
    export_data = []
    for block, text in results:
        export_data.append({
            "type": block.type,
            "score": float(block.score) if block.score is not None else None,
            "coordinates": {
                "x1": float(block.block.x_1),
                "y1": float(block.block.y_1),
                "x2": float(block.block.x_2),
                "y2": float(block.block.y_2),
            },
            "text": text
        })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Exported results to {output_json}")

# --- Main pipeline ---
def process_newspaper_page(image_path: str, output_visual="layout_result.png",
                           output_json="layout_result.json", column_count=None):
    model = load_layout_model()
    image_pil, layout = detect_layout(image_path, model)
    columns = cluster_into_columns(layout, n_columns=column_count)

    col_xcenters = [np.mean([(b.block.x_1 + b.block.x_2)/2 for b in col]) if len(col)>0 else np.inf for col in columns]
    sorted_columns = [c for _, c in sorted(zip(col_xcenters, columns), key=lambda x: x[0])]

    reading_order_blocks = lp.Layout()
    for col in sorted_columns:
        reading_order_blocks.extend(col)

    text_blocks = lp.Layout([b for b in reading_order_blocks if b.type in ("Text", "Title", "List")])
    ocr_results = ocr_layout_blocks(image_pil, text_blocks)

    ocr_map = []
    tb_index = 0
    for b in reading_order_blocks:
        if b.type in ("Text", "Title", "List"):
            ocr_map.append((b, ocr_results[tb_index][1]))
            tb_index += 1
        else:
            ocr_map.append((b, ""))

    visualize_layout(image_pil, reading_order_blocks, ocr_texts=ocr_map, save_path=output_visual)
    save_as_json(ocr_map, output_json)
    return ocr_map

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect newspaper layout + OCR text blocks using doctr, export JSON")
    parser.add_argument("image", help="Path to input newspaper page image")
    parser.add_argument("--out", default="layout_result.png", help="Path to save visualization")
    parser.add_argument("--json", default="layout_result.json", help="Path to save JSON output")
    parser.add_argument("--columns", type=int, default=None, help="Force number of columns")
    args = parser.parse_args()

    results = process_newspaper_page(args.image, args.out, args.json, args.columns)
    for i, (block, text) in enumerate(results):
        if text:
            print(f"\n---- Block {i} ({block.type}) ----\n{text[:300]}\n")
