"""Controle qualite des annotations (bounding boxes) du dataset EPI, et correction
des boites erronees quand c'est possible (§4.1.1 et §4.2 du sujet).

Erreurs recherchees :
- coordonnees hors [0,1] (annotation mal normalisee / bbox qui deborde de l'image)
- boites degenerees (largeur ou hauteur quasi nulle -> aucune information utile)
- ratio d'aspect extreme (probable erreur d'annotation plutot qu'un objet reel)

Correction appliquee :
- coordonnees clippees dans [0,1]
- boites degenerees supprimees de l'annotation corrigee
- le reste des lignes (valides) est conserve tel quel

Les fichiers corriges sont ecrits dans un dossier separe (jamais d'ecrasement des
originaux), pour tracabilite.
"""
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EPI_DATASET_ROOT", str(SCRIPT_DIR.parent / "Applications" / "dataset_yolo_epi")))
OUT_LABELS = SCRIPT_DIR / "annotations_epi_corrigees"
CLASSES = {0: "helmet", 1: "head", 2: "safety-vest"}

DEGENERATE_EPS = 0.002       # largeur/hauteur normalisee en dessous de laquelle une boite est jugee degeneree
ASPECT_RATIO_MAX = 15.0      # ratio largeur/hauteur (ou inverse) au-dela duquel on flag l'annotation comme suspecte

report = {
    "n_files_total": 0,
    "n_boxes_total": 0,
    "n_boxes_out_of_range": 0,
    "n_boxes_degenerate_removed": 0,
    "n_boxes_extreme_aspect_ratio_flagged": 0,
    "n_files_corrected": 0,
    "examples_out_of_range": [],
    "examples_degenerate": [],
    "examples_extreme_aspect_ratio": [],
    "per_class_box_count": {v: 0 for v in CLASSES.values()},
}

for split in ("train", "val", "test"):
    lbl_dir = ROOT / "labels" / split
    out_dir = OUT_LABELS / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        report["n_files_total"] += 1
        corrected_lines = []
        file_changed = False

        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            report["n_boxes_total"] += 1
            report["per_class_box_count"][CLASSES.get(cls, f"unknown_{cls}")] = (
                report["per_class_box_count"].get(CLASSES.get(cls, f"unknown_{cls}"), 0) + 1
            )

            # 1) coordonnees hors [0,1] -> clip
            raw = (xc, yc, bw, bh)
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            out_of_range = x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1
            if out_of_range:
                report["n_boxes_out_of_range"] += 1
                if len(report["examples_out_of_range"]) < 10:
                    report["examples_out_of_range"].append(f"{lbl_path.relative_to(ROOT)}: {raw}")
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
                xc, yc, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                file_changed = True

            # 2) boite degeneree -> suppression
            if bw <= DEGENERATE_EPS or bh <= DEGENERATE_EPS:
                report["n_boxes_degenerate_removed"] += 1
                if len(report["examples_degenerate"]) < 10:
                    report["examples_degenerate"].append(f"{lbl_path.relative_to(ROOT)}: w={bw:.4f} h={bh:.4f}")
                file_changed = True
                continue  # ligne supprimee de l'annotation corrigee

            # 3) ratio d'aspect extreme -> signale (conserve, car pas toujours une erreur
            #    reelle : une bande de securite tres fine et longue est plausible)
            ratio = bw / bh if bh > 0 else float("inf")
            if ratio > ASPECT_RATIO_MAX or ratio < 1 / ASPECT_RATIO_MAX:
                report["n_boxes_extreme_aspect_ratio_flagged"] += 1
                if len(report["examples_extreme_aspect_ratio"]) < 10:
                    report["examples_extreme_aspect_ratio"].append(
                        f"{lbl_path.relative_to(ROOT)}: ratio={ratio:.1f} (w={bw:.3f}, h={bh:.3f})"
                    )

            corrected_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if file_changed:
            report["n_files_corrected"] += 1
        (out_dir / lbl_path.name).write_text("\n".join(corrected_lines) + ("\n" if corrected_lines else ""))

out_json = SCRIPT_DIR / "annotation_quality_report.json"
with open(out_json, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps({k: v for k, v in report.items() if not k.startswith("examples")}, indent=2))
print(f"\nRapport complet : {out_json}")
