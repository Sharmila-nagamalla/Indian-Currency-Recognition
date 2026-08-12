@echo off

echo Creating Indian-Currency-Recognition-TensorFlow

:: Project folders
mkdir Indian-Currency-Recognition-TensorFlow
cd Indian-Currency-Recognition-TensorFlow

mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\split
mkdir data\split\train
mkdir data\split\validation
mkdir data\split\test

mkdir notebooks

mkdir src

mkdir models

mkdir outputs
mkdir outputs\graphs

mkdir app


:: Create source files
type nul > src\__init__.py
type nul > src\config.py
type nul > src\data_loader.py
type nul > src\transforms.py
type nul > src\model.py
type nul > src\train.py
type nul > src\evaluate.py
type nul > src\predict.py
type nul > src\utils.py


:: Notebook
type nul > notebooks\data_analysis.ipynb


:: Application
type nul > app\streamlit_app.py


:: Root files
type nul > main.py
type nul > requirements.txt
type nul > README.md
type nul > .gitignore


:: Model placeholder
type nul > models\best_currency_model.pth


:: Output placeholders
type nul > outputs\confusion_matrix.png
type nul > outputs\graphs\loss_curve.png
type nul > outputs\graphs\accuracy_curve.png


echo.
echo =====================================
echo Project structure created successfully
echo =====================================

