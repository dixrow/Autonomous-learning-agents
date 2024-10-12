# Autonomous Agents for Education

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange)

## Overview

This project is designed to simulate interactions between a teacher and a student using machine learning models. The goal is to create an educational environment where a student model can learn from a teacher model through a series of question-and-answer sessions.

This project was developed in collaboration with [atslisbona](https://github.com/atslisbona) as part of my final master's project.

## Features

- **Teacher and Student Models**: Utilizes transformer models to simulate educational interactions.
- **Multi-turn and Single-turn Interactions**: Supports both multi-turn and single-turn question-answering sessions.
- **Data Transformation**: Processes and transforms educational data for model training and evaluation.

## Technologies Used

- **Python**: The primary programming language used for development.
- **Transformers**: Utilizes the Hugging Face Transformers library for model implementation.
- **PyTorch**: Used for model training and inference.
- **Pandas**: For data manipulation and analysis.

## Setup Instructions

1. **Clone the Repository**:
   ```shell
   git clone https://github.com/dixrow/autonomous-learning-agents.git
   cd autonomous-learning-agents
   ```

2. **Create and Activate a Virtual Environment**:
   ```shell
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```shell
   pip install -r requirements.txt
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configure Jupyter Kernel**:
   ```shell
   python -m install ipykernel --user --name ALA
   ```

5. **Run the Jupyter Notebook**:
   Open the `tfm_transformers.ipynb` notebook and execute the cells to start the interaction simulations.

## Usage

- **Student Model**: Implemented in `Student.py` to simulate a student's learning process.
- **Teacher Model**: Implemented in `Teacher.py` to simulate a teacher's guidance.
- **Interactions**: Managed in `Interactions.py` to handle the flow of questions and answers.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [Your Name] at [Your Email].
