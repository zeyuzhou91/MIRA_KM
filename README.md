# MIRA_KM

**MIRA_KM** is a Python-based research toolkit designed for PET/MR image processing and modeling, particularly focused on the analysis of cerebrospinal fluid (CSF) dynamics in the human brain using FDG-PET imaging. It provides modular tools for segmentation, registration, kinetic modeling, and statistical evaluation.

## Features

- 🧠 **Neuroimaging Integration**: Built to work seamlessly with FreeSurfer and FSL for anatomical processing, co-registration, and transformation.
- 📈 **Kinetic Modeling**: Includes gamma-variate modeling and time-activity curve fitting for PET quantification.
- 📦 **Modular Codebase**: Clean and extendable code organized under `src/mira_km`.
- 🔬 **Statistical Tools**: Includes tools for computing ICC, Bland-Altman plots, and other reliability analyses.
- 🧪 **Test Scripts**: Includes runnable scripts for example datasets and development testing in `tests/`.

## Project Structure

```
MIRA_KM/
├── src/
│   └── mira_km/
│       ├── kinetic_modeling/     # PET modeling functions
│       ├── segmentation/         # Ventricular segmentation logic
│       ├── registration/         # Co-registration and transforms
│       ├── utils/                # Shared helper functions
│       ├── subject.py            # Subject and cohort abstractions
│       └── frameschedule.py      # Frame schedule parsing and modeling
├── tests/
│   └── FDG_PET_processing/
│       └── FDG_PET_processing.py # Example/test pipeline for PET data
├── data/                         # Placeholder for raw and processed data (not tracked by git)
└── README.md                     # Project overview (this file)
```

## Installation

Clone the repository:

```bash
git clone https://github.com/zeyuzhou91/MIRA_KM.git
cd MIRA_KM
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Ensure that external neuroimaging software (FreeSurfer, FSL) is properly installed and configured.

## Usage

You can execute a processing script from the `tests` folder, such as:

```bash
python tests/FDG_PET_processing/FDG_PET_processing.py
```

Be sure to adjust file paths and environmental variables to match your local setup, especially:

- `FREESURFER_HOME`
- `SUBJECTS_DIR`
- Data paths (NIfTI, LTA, segmentation masks, etc.)

## Requirements

- Python 3.10+
- FreeSurfer
- FSL
- Python packages:
  - `numpy`
  - `scipy`
  - `nibabel`
  - `pandas`
  - `matplotlib`
  - `pingouin`

## Citation

If you use this software in your work, please cite the related publications (citation to be added when available).

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

Maintainer: [Zeyu Zhou](https://github.com/zeyuzhou91)  
For questions, please open an issue or contact the maintainer directly.

---

_This project is part of ongoing research on neurofluid imaging and CSF dynamics. Contributions and collaborations are welcome._