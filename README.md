# MIRA_KM

**MIRA_KM** (Medical Image Research and Analysis - Kinetic Modeling) is a Python framework for preprocessing, modeling, and analysis of dynamic PET/MR neuroimaging data. It provides modular tools for researchers to develop and test image processing pipelines with support for PET kinetic modeling and anatomical integration.

## Features

- PET/MR image processing pipelines
- Frame scheduling and kinetic modeling support
- Integration with FreeSurfer and FSL for structural segmentation and registration
- Utilities for motion correction, segmentation, and spatial transformations

## Repository Structure

```
MIRA_KM/
├── docs/                            # Project documentation, focused on kinetic models
├── models/                          # Kinetic models
├── utils/                           # Utility modules and functions
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/zeyuzhou91/MIRA_KM.git
cd MIRA_KM
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure FreeSurfer and FSL are installed and correctly configured in your environment.

## Usage

Import modules from `mira_km` to construct and run your image processing pipelines. Refer to the scripts and notebooks for detailed usage examples.

Example:
```python
from mira_km.subject import SubjectRegistry
from mira_km.frameschedule import FrameSchedule
```


## License

This project is licensed under the MIT License.


## Acknowledgments

Developed by the MIRA research team at Emory University, with applications in CSF dynamics and PET kinetic modeling.
