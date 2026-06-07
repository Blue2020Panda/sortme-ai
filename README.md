# SortMeAI

SortMeAI is a Streamlit app that sorts photos into folders with YOLOv8 object detection. Upload individual JPG/PNG files or a ZIP archive of a folder, choose the objects you care about, and the app copies each image into matching output folders.

## Features

- Sorts individual images or all JPG/PNG files inside an uploaded ZIP archive.
- Uses YOLOv8 Open Images V7 detection classes.
- Scores detections with confidence, object size, and distance from the image center.
- Handles duplicate filenames, unreadable images, small images, and ZIP subfolders.
- Creates a downloadable `sorted_photos.zip` archive after sorting.

## Requirements

- Python 3.10+
- A `classes.txt` file in the project root for the category selector.
- Internet access on first model load so Hugging Face can download the YOLO model.

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The App

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

On first run, the app downloads `yolov8x-oiv7.pt` from the Hugging Face repo `aldencm/image-detector`. Streamlit caches the loaded model for the session.

## Usage

1. Start the Streamlit app.
2. In the sidebar, choose `Individual images` or `Folder as ZIP`.
3. Upload JPG/PNG files, or upload a ZIP containing JPG/PNG files.
4. Select one or more Open Images V7 category names.
5. Click `Sort My Photos`.
6. Review detections, scores, folder assignments, and explanations.
7. Download the generated `sorted_photos.zip` file.

## Scoring

Each detection receives a final score:

```text
final_score = confidence * 0.50 + size_score * 0.30 + center_score * 0.20
```

| Term | Meaning |
| --- | --- |
| `confidence` | YOLO detection confidence from 0 to 1. |
| `size_score` | Bounding box area divided by total image area. |
| `center_score` | How close the detected object is to the image center. |

Routing thresholds:

- `score >= 0.60`: copied into the matching category folder.
- `0.45 <= score < 0.60`: copied into `Needs_Review` if there is no confirmed category.
- `score < 0.45`, no matching category, or no detections: copied into `Unsorted`.

An image can be copied into multiple category folders when multiple selected objects score highly enough.

## Output

Each run clears and recreates the local `sorted_photos/` directory:

```text
sorted_photos/
  Dog/
  Person/
  Needs_Review/
  Unsorted/
```

Original uploaded files are not modified. SortMeAI writes copies into the output folders and packages that folder as a ZIP for download.

## Category Aliases

The app expands some common categories to related Open Images labels before matching detections.

| Category | Example labels matched |
| --- | --- |
| `alligator` | alligator, crocodile |
| `person` | person, man, woman, boy, girl, human body, human face |
| `car` | car, truck, bus, van, taxi, ambulance, limousine |
| `vehicle` | car, truck, bus, motorcycle, bicycle, airplane, boat, train |
| `dog` | dog |
| `cat` | cat |
| `food` | pizza, sandwich, hamburger, cake, pastry, sushi, pasta |
| `beach` | surfboard, swimwear, swimming pool |
| `bird` | bird, eagle, parrot, owl, penguin, duck |
| `tree` | tree, palm tree, flower, houseplant |
| `turtle` | turtle, tortoise, sea turtle |
| `bear` | bear, polar bear |

Categories without alias entries still match exact YOLO labels.

## Project Files

```text
app.py              Streamlit app, model loading, scoring, sorting, and ZIP export
requirements.txt    Python dependencies
README.md           Project documentation
.gitignore.example  Suggested ignore rules for local development
```

## Git Ignore Setup

Copy `.gitignore.example` to `.gitignore` when initializing a Git repository:

```bash
cp .gitignore.example .gitignore
```

The example keeps generated outputs, virtual environments, Python caches, Streamlit secrets, and downloaded model artifacts out of version control.

## License

MIT
