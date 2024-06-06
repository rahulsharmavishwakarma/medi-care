# Medi-Care Project

## Introduction
Medi-Care utilizes LLaVA-Med base weights for precise visual question answering in medical images, alongside the LLaMA model for accurate answer generation. This combination enables efficient and accurate diagnostic support, enhancing healthcare outcomes.

## Installation

### Steps

1. **Clone the repository:**
    ```sh
    git clone https://github.com/rahulsharmavishwakarma/medi-care.git
    cd medi-care
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Upgrade pip:**
    ```sh
    pip install --upgrade pip
    ```

4. **Install the required libraries:**
    Ensure you have a `requirements.txt` file and then run the following command to install all dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. **Install Git Large File Storage (LFS) if required:**
    ```sh
    git lfs install
    ```

## Starting the Inference

To start the inference, run the following commands in the terminal:

1. **Start the controller:**
    ```sh
    python -m llava.serve.controller --host 0.0.0.0 --port 10008
    ```

2. **Start the model worker:**
    ```sh
    python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10008 --port 40000 --worker http://localhost:40000 --model-path /teamspace/studios/this_studio/Medi-Care/LLaVA-Med-weights --multi-modal --num-gpus 1
    ```

3. **Run the Gradio web server:**
    ```sh
    python -m llava.serve.gradio_web_server --controller http://localhost:10008 --share
    ```

## Contributing
We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.


---

*This README was generated using a template. Make sure to customize it to fit your project's details and requirements.*
