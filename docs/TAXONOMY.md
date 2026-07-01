# Taxonomy

The initial taxonomy is defined in `configs/classes.yaml`. It separates labels into four top-level groups so the project can evolve into multiple specialized models while keeping one API response format.

## Crops

Crop labels identify the plant or commodity visible in the uploaded image. These labels help route the prediction to the most relevant pest or disease model and make recommendations crop-specific.

## Pests

Pest labels identify insects and related crop-damaging organisms. The first model iteration can treat these as classification labels, while later detection models may localize damage or visible pests with bounding boxes.

## Diseases

Disease labels identify visual plant health problems such as blight, rust, mildew, wilt, or viral symptoms. Disease labels should eventually include crop-specific mappings because the same symptom name can vary by crop.

## Animals

Animal labels identify field intrusion or crop damage risks from animals. These labels are separate from crop pests because they will likely use different datasets, camera viewpoints, and model assumptions.

## Maintenance

- Add labels only after confirming the use case and likely dataset availability.
- Keep common names readable for farmers and map scientific names in API responses when available.
- Avoid mixing crop labels with pest, disease, or animal problem labels.
- Record label changes in documentation before training models.
