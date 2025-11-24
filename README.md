# image-classifier
Classify an Image 

```
image_classifier/
│
├─ README.md
├─ requirements.txt
│
├─ notebooks/
│   └─ 01_explore_data.ipynb       # visualize dataset, shapes, sanity checks
│
├─ src/
│   ├─ config.py                   # hyperparameters
│   ├─ dataset.py                  # DataLoader + transforms
│   ├─ model.py                    # CNN class
│   ├─ train.py                    # training loop
│   ├─ evaluate.py                 # test accuracy + visuals
│   └─ utils.py                    # helper functions (accuracy, plotting)
│
└─ saved_models/
    └─ best_model.pth

```