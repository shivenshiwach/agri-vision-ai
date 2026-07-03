# Label Mapping Review

Milestone 7 records approved source-label to MVP-class mappings in `configs/label_mapping.yaml`. This milestone does not train models, move or copy dataset files, or approve any API behavior change.

## Approved PlantVillage Mappings

| Source label | MVP target |
| --- | --- |
| `Potato___Early_blight` | Early Blight |
| `Tomato___Early_blight` | Early Blight |
| `Potato___Late_blight` | Late Blight |
| `Tomato___Late_blight` | Late Blight |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Leaf Curl Virus |
| `Corn_(maize)___Common_rust_` | Rust |
| `Apple___Cedar_apple_rust` | Rust |
| `Squash___Powdery_mildew` | Powdery Mildew |
| `Cherry_(including_sour)___Powdery_mildew` | Powdery Mildew |

## Approved IP102 Mappings

| Source label | MVP target |
| --- | --- |
| `25 aphids` | Aphids |
| `55 Thrips` | Thrips |
| `13 grain spreader thrips` | Thrips |
| `23 corn borer` | Stem Borer |
| `24 army worm` | Fall Armyworm |
| `4 asiatic rice borer` | Stem Borer |
| `5 yellow rice borer` | Stem Borer |

## Rejected Weak Mappings

The following mappings are intentionally not approved:

- Whitefly from `white backed plant hopper`: rejected because a plant hopper is not a whitefly label.
- Bacterial Blight from PlantVillage bacterial spot labels: rejected because bacterial spot is not bacterial blight.
- Downy Mildew from non-exact mildew or blight labels: rejected until an exact downy mildew source label is identified and reviewed.

## MVP Classes Still Needing Data

The approved mappings cover only a subset of MVP pest and disease classes. The following MVP classes still have no approved source-label mapping in `configs/label_mapping.yaml`:

| Group | Classes still needing data |
| --- | --- |
| Crops | Cotton, Soybean, Sugarcane, Paddy (Rice), Wheat, Grapes, Pomegranate, Banana, Tomato, Onion |
| Pests | Pink Bollworm, Whitefly, Fruit Borer |
| Diseases | Downy Mildew, Bacterial Blight, Blast Disease |
| Animals | Wild Boar, Monkeys, Birds, Parrots, Sparrows, Rats, Snakes |

Mapped classes still require license review, duplicate checks, train/validation split design, and field-image validation before any model training.

## Validation

Run:

```bash
python scripts/validate_label_mapping.py
```
