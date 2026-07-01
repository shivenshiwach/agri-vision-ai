# Model Strategy

Milestone 1 defines the taxonomy and dataset plan only. No datasets are downloaded and no models are trained yet.

## Crop Classifier

The crop classifier should identify the visible crop or commodity before problem detection runs. This model can start as an image classifier using the crop classes in `configs/classes.yaml`, then later route images to crop-specific pest and disease models.

Initial focus should be high-volume and high-risk Maharashtra crops such as Cotton, Soybean, Sugarcane, Paddy (Rice), Wheat, Grapes, Pomegranate, Banana, Tomato, and Onion.

## Pest Detector

The pest detector should localize visible insects, larvae, and crop damage signs. Object detection is preferred over pure classification because many pest problems need bounding boxes, counts, and severity estimates.

The first target set should stay small: Pink Bollworm, Whitefly, Thrips, Aphids, Fruit Borer, Stem Borer, and Fall Armyworm. These classes are common, economically important, and likely to have at least partial public data.

## Disease Detector

The disease detector should identify visible leaf, stem, fruit, or storage symptoms. It can start with disease classification where bounding boxes are unavailable, then move to detection or segmentation for lesion-level severity.

Good first targets are Rust, Powdery Mildew, Downy Mildew, Early Blight, Late Blight, Bacterial Blight, Leaf Curl Virus, and Blast Disease.

## Animal Detector

The animal detector should identify crop-damaging wildlife, birds, rodents, and reptiles in field or storage images. Public wildlife datasets can support pretraining, but farm-context images will be required for deployment.

Initial classes should include Wild Boar, Monkeys, Birds, Parrots, Sparrows, Rats, and Snakes because these are visible, actionable, and appear in the regional mapping.

## Severity Estimator

Severity estimation should be added after crop, pest, disease, and animal baselines are stable. It should use model outputs plus agronomic thresholds from the project document, such as larvae per plant, percent infected leaves, dead-heart percentage, flock feeding, or repeated field entry.

Severity should begin with simple bands: low, medium, high, and severe. Later versions can use segmentation area, object counts, crop stage, and farmer feedback to improve accuracy.
