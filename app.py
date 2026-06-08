"""
SortMeAI - AI Photo Organizer
Uses YOLOv8 object detection to automatically sort photos into user-defined folders.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Alias groups: maps a canonical name to a list of Open Images V7 labels (and synonyms)
# that count as a match for that category.
# Keys are lowercase. Labels must match OIV7 class names (case-insensitive).
ALIASES: dict[str, list[str]] = {
    "alligator": ["alligator", "crocodile"],
    "person":    ["person", "man", "woman", "boy", "girl",
                  "human body", "human face", "human hand", "human arm", "human leg"],
    "car":       ["car", "truck", "bus", "van", "taxi", "ambulance", "limousine"],
    "vehicle":   ["car", "truck", "bus", "van", "taxi", "ambulance", "limousine",
                  "motorcycle", "bicycle", "airplane", "boat", "train", "helicopter"],
    "dog":       ["dog"],
    "cat":       ["cat"],
    "food":      ["pizza", "sandwich", "submarine sandwich", "hamburger", "hot dog",
                  "cake", "pastry", "cookie", "bread", "muffin", "waffle", "pancake",
                  "croissant", "pretzel", "tart", "burrito", "sushi", "pasta",
                  "salad", "french fries", "popcorn", "donut"],
    "beach":     ["surfboard", "swimwear", "swimming pool"],
    "bird":      ["bird", "eagle", "parrot", "owl", "penguin", "duck",
                  "blue jay", "magpie", "sparrow", "ostrich", "falcon"],
    "tree":      ["tree", "palm tree", "christmas tree", "flower", "houseplant",
                  "rose", "sunflower"],
    "turtle":    ["turtle", "tortoise", "sea turtle"],
    "bear":      ["bear", "polar bear"],
}

# Scoring thresholds — tweak these to change how strict the sorter is
SCORE_HIGH = 0.60   # score >= SCORE_HIGH → confirmed category
SCORE_LOW  = 0.45   # SCORE_LOW <= score < SCORE_HIGH → Needs Review

OUTPUT_ROOT = "sorted_photos"   # top-level output directory name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# ZIP Folder Extraction
# ---------------------------------------------------------------------------

def extract_images_from_zip(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Walk a ZIP archive and return (filename, raw_bytes) for every image inside.
    Sub-folder paths are flattened into the filename with underscores so that
    dogs/fido.jpg and cats/fido.jpg become dogs_fido.jpg / cats_fido.jpg,
    avoiding silent collisions when the deduplicator runs later.
    """
    items: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.infolist():
                # Skip directories and macOS/Windows metadata entries
                if member.filename.endswith("/"):
                    continue
                basename = os.path.basename(member.filename)
                if basename.startswith(".") or basename.startswith("__"):
                    continue
                if Path(basename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # Flatten subfolder path: "vacation/dogs/fido.jpg" → "vacation_dogs_fido.jpg"
                flat_name = member.filename.replace("\\", "/").strip("/").replace("/", "_")
                try:
                    items.append((flat_name, zf.read(member.filename)))
                except Exception as e:
                    st.warning(f"Could not read {member.filename} from ZIP: {e}")
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    return items


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_yolo_model() -> YOLO:
    """
    Load the YOLOv8n-OIV7 model (Open Images V7, 600 classes) once and cache it.
    The first time this runs, ultralytics downloads yolov8n-oiv7.pt automatically (~12MB).
    @st.cache_resource means Streamlit only runs this function once per session,
    so we don't reload the model every time the page refreshes.
    """
    try:
        model_path = hf_hub_download(
            repo_id="Blue2020Panda/YOLOv8mOIV7",
            filename="yolov8m-oiv7.pt",
        )
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Class List Loader
# ---------------------------------------------------------------------------

@st.cache_data
def load_oiv7_classes(filepath: str = "SortMeAI/classes.txt") -> list[str]:
    classes = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if ": " in line:
                    classes.append(line.split(": ", 1)[1])
    except FileNotFoundError:
        st.error(f"classes.txt not found at {filepath}")
    return classes


# ---------------------------------------------------------------------------
# Scoring Helpers
# ---------------------------------------------------------------------------

def compute_size_score(x1: float, y1: float, x2: float, y2: float,
                       img_w: int, img_h: int) -> float:
    """
    How large is the detected object relative to the whole image?
    Returns a value between 0.0 (tiny) and 1.0 (fills the entire image).
    Formula: bbox_area / total_image_area
    """
    total_area = img_w * img_h
    if total_area == 0:
        return 0.0
    bbox_area = (x2 - x1) * (y2 - y1)
    return float(min(1.0, max(0.0, bbox_area / total_area)))


def compute_center_score(x1: float, y1: float, x2: float, y2: float,
                         img_w: int, img_h: int) -> float:
    """
    How close is the detected object to the center of the image?
    Returns 1.0 if perfectly centered, 0.0 if at the furthest corner.
    Formula: 1 - (distance_to_center / max_possible_distance)
    """
    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    if max_dist == 0:
        return 1.0
    # Center of the bounding box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # Distance from bbox center to image center
    distance = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
    return float(min(1.0, max(0.0, 1.0 - (distance / max_dist))))


def compute_final_score(confidence: float, size_score: float,
                        center_score: float) -> float:
    """
    Combine the three signals into one final score using a weighted formula:
      final = confidence × 0.50 + size_score × 0.30 + center_score × 0.20
    Confidence matters most (50%), size is secondary (30%), centering is tertiary (20%).
    """
    return confidence * 0.50 + size_score * 0.30 + center_score * 0.20


# ---------------------------------------------------------------------------
# Detection Pipeline
# ---------------------------------------------------------------------------

def run_detection(model: YOLO, image_bytes: bytes) -> tuple[list[dict], tuple[int, int]]:
    """
    Run YOLOv8 on a single image and return a list of Detection dicts plus the image size.

    Each Detection dict has:
        label, confidence, bbox (x1,y1,x2,y2), size_score, center_score, final_score

    Returns ([], (0, 0)) if detection fails for any reason.
    """
    try:
        # Open the raw bytes as a PIL image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_image.size

        # YOLO can crash or produce bad results on very small images
        if img_w < 32 or img_h < 32:
            st.warning(f"Image too small for detection ({img_w}×{img_h}px). Placed in Unsorted.")
            return [], (img_w, img_h)

        # Run inference — verbose=False suppresses YOLO's console output
        # Convert to numpy array explicitly; passing a PIL image directly can
        # trigger "Numpy is not available" inside YOLO's internal conversion.
        results = model(np.array(pil_image), verbose=False)

        detections: list[dict] = []

        # results is a list of one Results object (one image)
        if not results or results[0].boxes is None:
            return detections, (img_w, img_h)

        for box in results[0].boxes:
            # Extract label, confidence, and bounding box coordinates
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Compute the three component scores
            size_score   = compute_size_score(x1, y1, x2, y2, img_w, img_h)
            center_score = compute_center_score(x1, y1, x2, y2, img_w, img_h)
            final_score  = compute_final_score(confidence, size_score, center_score)

            detections.append({
                "label":        label,
                "confidence":   confidence,
                "bbox":         (int(x1), int(y1), int(x2), int(y2)),
                "size_score":   size_score,
                "center_score": center_score,
                "final_score":  final_score,
            })

        return detections, (img_w, img_h)

    except UnidentifiedImageError:
        st.warning("Could not read one image file — it may be corrupt. Placed in Unsorted.")
        return [], (0, 0)
    except Exception as e:
        st.warning(f"Detection failed for an image: {e}")
        return [], (0, 0)


# ---------------------------------------------------------------------------
# Category Matching & Routing
# ---------------------------------------------------------------------------

def build_category_lookup(user_categories: list[str]) -> dict[str, str]:
    """
    Pre-compute a flat lookup table: {yolo_label_or_alias: canonical_user_category}

    This runs once before processing all images so that per-image matching is O(1).

    Example: user types "dog, beach"
        lookup["dog"]       = "dog"
        lookup["puppy"]     = "dog"
        lookup["beach"]     = "beach"
        lookup["ocean"]     = "beach"
        lookup["water"]     = "beach"
        ...
    """
    lookup: dict[str, str] = {}

    # Pass 1: lock in each user category as an exact self-mapping so these
    # cannot be overwritten by alias expansion in pass 2.
    for user_cat in user_categories:
        cat = user_cat.lower().strip()
        if cat:
            lookup[cat] = cat

    # Pass 2: expand alias groups, but never overwrite an exact match from pass 1.
    for user_cat in user_categories:
        cat = user_cat.lower().strip()
        if not cat:
            continue

        if cat in ALIASES:
            for alias in ALIASES[cat]:
                if alias not in lookup:
                    lookup[alias] = cat
        else:
            for canonical, aliases in ALIASES.items():
                if cat in aliases:
                    for alias in aliases:
                        if alias not in lookup:
                            lookup[alias] = cat
                    if canonical not in lookup:
                        lookup[canonical] = cat
                    break

    return lookup


def assign_folders(detections: list[dict],
                   lookup: dict[str, str]) -> tuple[list[str], str]:
    """
    Decide which output folders an image belongs in based on its detections and scores.

    Returns:
        folders     - list of folder names (e.g. ["Dog", "Needs_Review"])
        explanation - human-readable sentence describing the decision
    """
    if not detections:
        return ["Unsorted"], "No objects detected. Placed in Unsorted."

    confirmed: list[str] = []   # folders where score >= SCORE_HIGH
    needs_review = False        # any detection in the borderline range?
    best_detection: dict | None = None
    best_score = -1.0
    for det in detections:
        label_lower = det["label"].lower()
        matched_cat = lookup.get(label_lower)
        score = det["final_score"]

        # Track the single best-scoring matched detection for the explanation
        if matched_cat and score > best_score:
            best_score = score
            best_detection = {**det, "matched_cat": matched_cat}

        if matched_cat:
            if score >= SCORE_HIGH:
                # Capitalize the folder name to look nice on disk
                folder = matched_cat.capitalize()
                if folder not in confirmed:
                    confirmed.append(folder)
            elif score >= SCORE_LOW:
                needs_review = True

    # Build the final folder list
    folders: list[str] = []
    folders.extend(confirmed)
    if needs_review and not confirmed:
        # Only add Needs_Review if the image didn't qualify for a confirmed folder
        folders.append("Needs_Review")
    if not folders:
        folders.append("Unsorted")

    # Build an explanation using the best-scoring matched detection
    if best_detection:
        size_desc   = "large"       if best_detection["size_score"]   > 0.5 else "small"
        center_desc = "centered"    if best_detection["center_score"] > 0.7 else "off-center"
        folder_list = ", ".join(folders)
        explanation = (
            f"Placed in {folder_list} because YOLO detected "
            f"{best_detection['label']} with "
            f"{best_detection['confidence']:.0%} confidence, "
            f"{size_desc} object size, and {center_desc} position."
        )
    elif needs_review:
        explanation = (
            "Scores were borderline (0.45–0.60). Placed in Needs_Review for manual review."
        )
    else:
        # Detections exist but none matched any user category
        detected_labels = ", ".join(set(d["label"] for d in detections))
        explanation = (
            f"No user categories matched the detected objects ({detected_labels}). "
            "Placed in Unsorted."
        )
    print("*" * 50)
    print(best_detection)
    return folders, explanation


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def deduplicate_filename(filename: str, seen_names: set[str]) -> str:
    """
    If `filename` already exists in seen_names, append _1, _2, ... to the stem.
    Example: "photo.jpg" → "photo_1.jpg" → "photo_2.jpg"
    """
    if filename not in seen_names:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in seen_names:
            return candidate
        counter += 1


def save_image_to_folders(image_bytes: bytes, filename: str,
                          folders: list[str]) -> list[str]:
    """
    Copy the raw image bytes into each folder under OUTPUT_ROOT.
    Returns the list of file paths that were written.
    Original bytes are written directly — no re-encoding, no quality loss.
    """
    saved_paths: list[str] = []
    for folder in folders:
        folder_path = os.path.join(OUTPUT_ROOT, folder)
        try:
            os.makedirs(folder_path, exist_ok=True)
            dest = os.path.join(folder_path, filename)
            with open(dest, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(dest)
        except OSError as e:
            st.warning(f"Could not save {filename} to {folder}: {e}")
    return saved_paths


def create_zip_archive(source_dir: str) -> bytes:
    """
    Compress the entire sorted_photos directory into a ZIP and return it as bytes.
    The bytes are passed directly to st.download_button — no temp file needed.
    """
    buffer = io.BytesIO()
    parent = os.path.dirname(os.path.abspath(source_dir))
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, start=parent)
                zf.write(abs_path, arcname=arc_name)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# UI Rendering Helpers
# ---------------------------------------------------------------------------

def render_summary_metrics(results: list[dict]) -> None:
    """
    Show a one-row dashboard with four metrics after processing completes.
    """
    total = len(results)
    sorted_count = sum(
        1 for r in results
        if any(f not in ("Needs_Review", "Unsorted") for f in r["assignments"])
    )
    needs_review_count = sum(1 for r in results if "Needs_Review" in r["assignments"])
    unsorted_count     = sum(1 for r in results if "Unsorted"     in r["assignments"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images",  total)
    col2.metric("Sorted",        sorted_count)
    col3.metric("Needs Review",  needs_review_count)
    col4.metric("Unsorted",      unsorted_count)


def render_image_result(result: dict, index: int) -> None:
    """
    Render one image's results inside a collapsible expander.
    Left column: image thumbnail. Right column: detections, scores, folders, explanation.
    """
    folder_label = ", ".join(result["assignments"])
    with st.expander(f"{index + 1}. {result['filename']}  →  {folder_label}"):
        left, right = st.columns([1, 2])

        with left:
            st.image(result["image"], width=300)

        with right:
            if result["detections"]:
                st.markdown("**Detected Objects**")
                # Build a list of rows for the scores table
                rows = []
                for det in result["detections"]:
                    rows.append({
                        "Object":       det["label"],
                        "Confidence":   round(det["confidence"],   3),
                        "Size Score":   round(det["size_score"],   3),
                        "Center Score": round(det["center_score"], 3),
                        "Final Score":  round(det["final_score"],  3),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.markdown("*No objects detected.*")

            st.markdown(f"**Folders:** `{folder_label}`")
            st.info(result["explanation"])


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    # Must be the very first Streamlit call in the script
    st.set_page_config(
        page_title="SortMeAI",
        page_icon="🗂️",
        layout="wide",
    )

    st.title("SortMeAI — AI Photo Organizer")
    st.markdown(
        "Upload photos, enter category names, and the AI will sort them into folders "
        "using YOLOv8 object detection (Open Images V7 — 600 classes)."
    )

    # Load the YOLO model (cached — only downloads/loads once per session)
    model = load_yolo_model()
    oiv7_classes = load_oiv7_classes()

    # -----------------------------------------------------------------------
    # Sidebar — inputs
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")

        upload_mode = st.radio(
            "Upload mode",
            ["Individual images", "Folder as ZIP"],
            horizontal=True,
            help=(
                "Individual: pick specific JPG/PNG files. "
                "Folder as ZIP: compress a whole folder into a .zip and upload it."
            ),
        )

        if upload_mode == "Individual images":
            if "uploader_key" not in st.session_state:
                st.session_state["uploader_key"] = 0

            uploaded_files = st.file_uploader(
                "Upload photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Select one or more JPG or PNG photos.",
                key=f"uploader_{st.session_state['uploader_key']}",
            )

            if uploaded_files:
                if st.button("Clear all photos", use_container_width=True):
                    st.session_state["uploader_key"] += 1
                    st.rerun()

            zip_file = None
        else:
            zip_file = st.file_uploader(
                "Upload folder as ZIP",
                type=["zip"],
                help=(
                    "Zip your image folder and upload it here. "
                    "All JPG/PNG files inside (including sub-folders) will be processed."
                ),
            )
            uploaded_files = None

        selected_categories = st.multiselect(
            "Category names",
            options=oiv7_classes,
            help="Type to search and select OIV7 object classes. The AI will sort photos into these folders.",
        )

        sort_button = st.button("Sort My Photos", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Processing — runs when the Sort button is clicked
    # -----------------------------------------------------------------------
    if sort_button:
        # --- Collect images from whichever upload mode was used ---
        image_items: list[tuple[str, bytes]] = []

        if upload_mode == "Individual images":
            if not uploaded_files:
                st.warning("Please upload at least one image before sorting.")
                st.stop()
            for f in uploaded_files:
                image_items.append((f.name, f.read()))
        else:
            if not zip_file:
                st.warning("Please upload a ZIP file before sorting.")
                st.stop()
            image_items = extract_images_from_zip(zip_file.read())
            if not image_items:
                st.warning("No JPG or PNG images were found inside the ZIP file.")
                st.stop()

        categories = [c.lower() for c in selected_categories]
        if not categories:
            st.warning("Please select at least one category.")
            st.stop()

        # --- Setup ---
        lookup = build_category_lookup(categories)

        # Clear the output directory so each Sort run starts fresh
        if os.path.exists(OUTPUT_ROOT):
            shutil.rmtree(OUTPUT_ROOT)
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        results: list[dict] = []
        seen_filenames: set[str] = set()
        total = len(image_items)

        # --- Processing loop ---
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, (orig_name, image_bytes) in enumerate(image_items):
            status_text.text(f"Processing {orig_name} ({i + 1}/{total})...")

            # Deduplicate filename in case of collisions
            safe_name = deduplicate_filename(orig_name, seen_filenames)
            seen_filenames.add(safe_name)

            # Run object detection
            detections, _ = run_detection(model, image_bytes)

            # Decide which folders this image belongs in
            folders, explanation = assign_folders(detections, lookup)

            # Save image bytes to each assigned folder
            saved_paths = save_image_to_folders(image_bytes, safe_name, folders)

            # Open PIL image for display (kept in memory for the results view)
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                pil_image = None

            results.append({
                "filename":    safe_name,
                "image":       pil_image,
                "detections":  detections,
                "assignments": folders,
                "explanation": explanation,
                "saved_paths": saved_paths,
            })

            progress_bar.progress((i + 1) / total)

        status_text.text(f"Done! Processed {total} image{'s' if total != 1 else ''}.")

        # Store results so they survive Streamlit reruns (e.g. clicking Download)
        st.session_state["results"] = results

    # -----------------------------------------------------------------------
    # Results display — shown after processing or on rerun
    # -----------------------------------------------------------------------
    results = st.session_state.get("results")

    if results:
        st.divider()
        st.subheader("Results")
        render_summary_metrics(results)

        st.markdown("---")
        for i, result in enumerate(results):
            render_image_result(result, i)

        # -----------------------------------------------------------------------
        # Download button
        # -----------------------------------------------------------------------
        st.divider()
        if os.path.exists(OUTPUT_ROOT):
            zip_bytes = create_zip_archive(OUTPUT_ROOT)
            st.download_button(
                label="Download sorted_photos as ZIP",
                data=zip_bytes,
                file_name="sorted_photos.zip",
                mime="application/zip",
                type="primary",
            )
    else:
        # Show a friendly placeholder when no results exist yet
        st.info(
            "Upload photos in the sidebar and click **Sort My Photos** to get started."
        )


if __name__ == "__main__":
    main()