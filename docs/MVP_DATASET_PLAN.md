# MVP Dataset Plan

Milestone 2 prepares dataset collection and preprocessing only. No datasets are downloaded and no models are trained in this milestone.

## Why Not Train All 144 Classes First

The full taxonomy has 144 classes across crops, pests, diseases, animals, birds, rodents, and reptiles. Training everything at once would create avoidable problems:

- Many classes have only partial or no public dataset coverage.
- Several labels need crop-specific context before they are useful, such as generic wilt, borer, blight, or bird labels.
- Pest and animal classes need bounding boxes, while crop classification can use image-level labels.
- A smaller MVP makes validation faster and reduces confusion between visually similar classes.
- Field data quality, class balance, and annotation rules need to be proven before scaling.

The MVP set uses high-priority Maharashtra crops and visible problems with better expected public coverage.

## MVP Dataset Sources

| Group | MVP Classes | Planned Sources |
| --- | --- | --- |
| Crops | Cotton, Soybean, Sugarcane, Paddy (Rice), Wheat, Grapes, Pomegranate, Banana, Tomato, Onion | PlantDoc, Roboflow, Open Images, PlantVillage for supported crops |
| Pests | Pink Bollworm, Whitefly, Thrips, Aphids, Fruit Borer, Stem Borer, Fall Armyworm | IP102, Roboflow, iNaturalist |
| Diseases | Rust, Powdery Mildew, Downy Mildew, Early Blight, Late Blight, Bacterial Blight, Leaf Curl Virus, Blast Disease | PlantVillage, PlantDoc, Roboflow |
| Animals | Wild Boar, Monkeys, Birds, Parrots, Sparrows, Rats, Snakes | iNaturalist, Open Images, Roboflow |

## Custom Field Data Needed

All MVP classes should receive custom Maharashtra field data before production use. Public datasets can support baselines, but they will not fully cover local crop varieties, farm backgrounds, growth stages, phone camera quality, mixed symptoms, and regional species.

Highest custom-data need:

- Pink Bollworm and Fruit Borer damage on Cotton, Tomato, and Pomegranate.
- Leaf Curl Virus on Cotton and Tomato.
- Blast Disease in Paddy (Rice) at field stage.
- Wild Boar, Monkeys, Sparrows, and Rats in real farm or storage scenes.
- Onion and Banana field images for crop routing and problem context.

## Expected Image Counts

Target collection size before first training:

| Model Type | Minimum per Class | Preferred per Class | Notes |
| --- | ---: | ---: | --- |
| Crop classifier | 500 | 1,500+ | Image-level labels are enough for the first baseline. |
| Pest detector | 300 annotated images | 1,000+ annotated images | Needs bounding boxes around visible insects or clear damage regions. |
| Disease detector | 300 annotated images | 1,000+ annotated images | Classification can start earlier, but detection needs symptom boxes. |
| Animal detector | 300 annotated images | 1,000+ annotated images | Include day/night, distance, occlusion, and field camera variation. |

Use balanced train, validation, and test splits. Do not mix near-duplicate images across splits.

## YOLO Annotation Format

YOLO expects one `.txt` label file per image. Each row represents one object:

```text
class_id x_center y_center width height
```

Rules:

- Coordinates are normalized from `0.0` to `1.0`.
- `class_id` must match the index in `configs/yolo_dataset_template.yaml`.
- Image files should live under `images/train`, `images/val`, and `images/test`.
- Label files should mirror the same relative names under `labels/train`, `labels/val`, and `labels/test`.
- Empty label files are allowed only when intentionally representing negative images, and should be reviewed before training.
