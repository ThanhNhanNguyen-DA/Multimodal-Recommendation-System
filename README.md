# Fashion Recommender System

This project implements a fashion recommender system that utilizes both text and image inputs to provide personalized recommendations. The system is designed to handle various fashion products and aims to enhance user experience by leveraging multimodal data.

## Project Structure

```
fashion-recommender
├── src
│   ├── build_recommender.py       # Main logic for building the recommender system
│   ├── app.py                     # Entry point for the application and API setup
│   ├── recommender.py             # Core functionality for generating recommendations
│   ├── data
│   │   ├── loader.py              # Functions for loading datasets
│   │   └── preprocess.py          # Functions for data preprocessing
│   ├── models
│   │   ├── text_encoder.py        # Class for encoding text inputs
│   │   ├── image_encoder.py       # Class for encoding image inputs
│   │   └── fusion.py              # Methods for fusing embeddings
│   ├── utils
│   │   ├── io.py                  # Utility functions for I/O operations
│   │   └── metrics.py             # Functions for evaluating performance
│   └── types
│       └── types.py               # Custom types and data structures
├── notebooks
│   └── exploration.ipynb          # Jupyter notebook for exploratory data analysis
├── tests
│   ├── test_recommender.py        # Unit tests for recommender functionality
│   └── test_utils.py              # Unit tests for utility functions
├── requirements.txt               # Python package dependencies
├── pyproject.toml                 # Project configuration and dependency management
├── .gitignore                     # Files and directories to ignore by version control
└── README.md                      # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd fashion-recommender
pip install -r requirements.txt
```

## Usage

1. **Building the Recommender**: Run the `build_recommender.py` script to load data, train models, and save artifacts.

   ```bash
   python src/build_recommender.py --data_path <path-to-data> --out_dir <output-directory>
   ```

2. **Running the Application**: Start the web server using `app.py` to expose the recommender system's API.

   ```bash
   python src/app.py
   ```

3. **Querying Recommendations**: Use the API endpoints defined in `app.py` to get recommendations based on text or image inputs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.