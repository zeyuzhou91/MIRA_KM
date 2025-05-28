# MIRA_KM

**MIRA_KM** (Modeling and Imaging of Regional Activity - Kinetic Modeling) is a Python framework for preprocessing, modeling, and analysis of dynamic PET/MR neuroimaging data, particularly focused on cerebrospinal fluid (CSF) dynamics. It provides modular tools for researchers to develop and test image processing pipelines with robust support for PET kinetic modeling and anatomical integration.

## Features

- End-to-end PET/MR image processing pipelines
- Frame scheduling and kinetic modeling support
- Integration with FreeSurfer and FSL for structural segmentation and registration
- Utilities for motion correction, segmentation, and spatial transformations
- Test-driven development framework and demonstration scripts

## Repository Structure

```
MIRA_KM/
├── configs/                         # Configuration files for pipelines
├── data/                            # Input and output data directory (empty or example structure)
├── docs/                            # Project documentation
├── notebooks/                       # Jupyter notebooks for examples and experiments
├── scripts/                         # CLI scripts to run pipelines and modules
├── src/
│   └── mira_km/
│       ├── analysis/                # Analysis and visualization tools
│       ├── frameschedule/           # Frame scheduling models for dynamic PET
│       ├── image/                   # Image utilities for registration and masking
│       ├── io/                      # Input/output helpers
│       ├── subject/                 # Subject-level data handling
│       ├── tools/                   # Interfaces to FreeSurfer, FSL, and external commands
│       ├── utils/                   # Utility functions (file handling, plotting, etc.)
│       └── workflow/                # Pipeline and task orchestration
├── tests/                           # Unit tests
│   └── FDG_PET_processing/          # FDG PET processing test script
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
from mira_km.subject import Cohort
from mira_km.frameschedule import FrameSchedule
```

To run a pipeline script:
```bash
python scripts/run_pipeline.py --config configs/pipeline_config.json
```

## Testing

To run the tests:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License.

## Contributions

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

## Acknowledgments

Developed by the neuroimaging research team at Emory University, with applications in CSF dynamics and PET kinetic modeling.
