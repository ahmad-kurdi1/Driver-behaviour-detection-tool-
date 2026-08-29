# Driver Behaviour Detection Tool

A computer vision application for detecting and analyzing driver behaviour using YOLO-based object detection and a Streamlit interface.

## Features

- Driver behaviour detection using YOLO
- Model training and testing scripts
- Result evaluation and chart generation
- Streamlit-based application interface
- Visualization of detection and performance results

## Project Structure

```text
.
├── app.py
├── config.py
├── full_test.py
├── generate_charts.py
├── main.py
├── train.py
├── requirements.txt
├── .gitignore
└── utils/
    ├── __init__.py
    └── helpers.py
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

## Training

Review the paths and settings in `config.py`, then run:

```bash
python train.py
```

## Testing

```bash
python full_test.py
```

## Notes

Large datasets, trained model weights, generated files, and Python cache files are intentionally excluded from the repository. Add the required dataset/model files locally according to the paths configured in `config.py`.

## Author

Ahmad Alaa
