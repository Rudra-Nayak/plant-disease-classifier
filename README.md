# Plant Disease Classifier (PlantGuard)
 
A ResNet-18 image classifier that identifies 38 plant disease and healthy classes across 14 crops from a leaf photo, served through a Streamlit web app.
 
![App screenshot](docs/screenshot.png)
 
## What it does
 
- Takes a leaf photo (file upload or camera) and returns the most likely disease with a confidence score.
- Shows the top 3 predictions so the user can see how close the alternatives are.
- States the model's limits in the interface, including that plants outside the 14 supported crops can get confident but wrong answers.
**Supported crops (14):** apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, tomato.
 
## Results
 
Evaluated on 1,500 images from the validation split of PlantVillage.
 
| Metric | Value |
| --- | --- |
| Validation accuracy | 89.47% |
| Weighted F1 | 0.90 |
| Macro F1 | 0.85 (about 0.88 if the empty Tomato healthy class is excluded) |
 
Full per-class results are in the classification report output, and the confusion matrix is in [`confusion_matrix.png`](confusion_matrix.png).
 
**Where it makes mistakes**
 
| True class | Predicted as | Count |
| --- | --- | --- |
| Tomato Late blight | Tomato Early blight | 12 |
| Tomato Yellow Leaf Curl Virus | Tomato Spider mites | 7 |
| Tomato Target Spot | Tomato healthy | 5 |
 
Most of the errors are among tomato diseases that look alike (Early blight F1 0.42, Late blight F1 0.69). The Target Spot to healthy errors are the most serious kind, because a diseased leaf is called healthy.
 
Note: the Tomato healthy class had no images in this validation run, so its row in the report is empty.
 
## How it works
 
1. **Data:** PlantVillage, 38 classes, 43,444 training images (one folder per class).
2. **Model:** ResNet-18 pretrained on ImageNet. The backbone is frozen and a new 38-class output layer is trained (transfer learning).
3. **Preprocessing:** images resized to 224x224 and normalized with ImageNet statistics, the same at training and prediction time.
4. **Evaluation:** per-class precision, recall and F1, plus a confusion matrix, on held-out validation images.
5. **App:** Streamlit loads `plant_model.pt` and `classes.json`, runs the model on the uploaded image and shows the top 3 predictions.
## Limitations
 
- **Only 14 crops.** The model must pick one of its 38 known classes, so it has no way to answer "none of these". Examples from testing: a rose leaf was labelled strawberry with 80 to 90% confidence, and an apple leaf with mite damage (a condition not in the training data) was labelled strawberry leaf scorch at 97.6%. A high confidence score means "closest known class", not "correct".
- **Lab-style training images.** PlantVillage photos show single leaves on plain backgrounds under even light. Real field photos will probably score lower than the numbers above, and I have not measured that gap yet.
- **Possible optimistic score.** PlantVillage contains several photos of the same leaf, so a random split can put near-duplicates in both training and validation.
- **Similar-looking diseases are confused,** especially tomato Early and Late blight.
## Quick start
 
1. Install dependencies:
```
   pip install -r requirements.txt
```
 
2. Get the dataset. Download PlantVillage and arrange it as `PlantVillage/train/<class>/` and `PlantVillage/val/<class>/`, one folder per class. The dataset is not included in this repo.
3. Train (optional if `plant_model.pt` and `classes.json` are already in the folder):
```
   python plant_disease_classifier.py --data_dir PlantVillage --epochs 10
```
 
   This produces `plant_model.pt` and `classes.json`. `fast_train.py` is a quicker CPU version of the same transfer-learning approach for testing.
 
4. Evaluate:
```
   python eval_report.py
```
 
   This prints the classification report and saves `confusion_matrix.png`.
 
5. Run the app:
```
   streamlit run app.py
```
 
   It opens at http://localhost:8501.
 
## Project structure
 
```
plant-disease-classifier/
  app.py                       Streamlit app
  plant_disease_classifier.py  Training script (ResNet-18, transfer learning)
  fast_train.py                Quick CPU training script
  eval_report.py               Classification report and confusion matrix
  confusion_matrix.png         Evaluation plot
  plant_model.pt               Trained weights
  classes.json                 Class names in model output order
  requirements.txt
  README.md
```
 
## Next steps
 
- Fine-tune the last ResNet block with a small learning rate and stronger augmentation, then compare against the current results.
- Add an "unknown" class and calibrate the confidence scores, so unsupported plants are less likely to get confident wrong answers.
- Build a small real-world test set of photos from outside PlantVillage to measure the gap between lab and field performance.
## Tech stack
 
Python, PyTorch, Torchvision, Scikit-learn, Matplotlib, NumPy, Pillow, Streamlit
 
## Dataset and credit
 
PlantVillage dataset: Mohanty, Hughes and Salathe, "Using Deep Learning for Image-Based Plant Disease Detection" (2016).
 








