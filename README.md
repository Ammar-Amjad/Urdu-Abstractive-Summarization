# Urdu Abstractive Summarization
This repository demonstrates the process of training our summarization model using PyTorch and the Transformers library. We further provide a step-by-step guide to preprocess a dataset, train our summarization model, and evaluate the model using ROUGE scores. More detail about our methodology and results can be seen in the [Abstractive Urdu Summarization Report](https://github.com/Ammar-Amjad/Urdu-Abstractive-Summarization/blob/main/Abstractive%20Urdu%20Summarization.pdf).

## Installation
To run the notebook, you need to install the required libraries. You can do this by running the following command:

In shell:
- pip install datasets evaluate transformers accelerate huggingface_hub rouge nltk

Install the required libraries using the command mentioned above.

Clone the repository and navigate to the notebook file.

Open the notebook in a Python environment that supports Jupyter notebooks.

Execute each cell in the notebook sequentially to run the code.

The notebook includes detailed explanations and comments to help you understand each step of the process. It covers tasks such as installing libraries, setting up the environment, loading the dataset, preprocessing the data, loading the model and tokenizer, training the model, and evaluating the results.

## Requirements
The notebook requires the following libraries:

- PyTorch
- Transformers
- Datasets
- Evaluate
- Accelerate
- Hugging Face Hub
- Rouge
- NLTK

## Approach:
The code is a Python notebook that trains a summarization model using PyTorch and Transformers. It performs the following steps:

1- Installs required libraries.

2- Sets up the environment.

3- Defines directories for dataset storage.

4- Loads the Urdu dataset.

5- Displays dataset samples.

6- Preprocesses the dataset by filtering and removing certain examples.

7- Loads a pre-trained model and tokenizer.

8- Defines a preprocessing function for tokenizing dataset examples.

9- Calculates ROUGE scores for evaluation.

11- Evaluates a baseline summarization approach.

12- Sets up training parameters.

13- Trains the model using the Seq2SeqTrainer.

14- Evaluates the trained model on the validation dataset.

Overall, the code showcases dataset preprocessing, model training, and evaluation for summarization using Transformers in PyTorch.

## Contributing
If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Credits
The notebook code and explanations were written by Ammar Amjad, Mohammad Anas, Micheal Perez and Mohammad Uzair. The code uses the Transformers library developed by Hugging Face.

## References
Transformers Documentation

Datasets Documentation

Accelerate Documentation

Hugging Face Hub Documentation

Rouge Library

NLTK Documentation

Please refer to the documentation of these libraries for more information on their usage and functionalities.

## Acknowledgements
We would like to acknowledge the developers and contributors of the PyTorch, Transformers, Datasets, Evaluate, Accelerate, Hugging Face Hub, Rouge, the model used for fine-tuning and NLTK libraries for their valuable work and contributions.

## Contact
For any inquiries or questions, please contact ammar_amjad@ymail.com or ammar.amjad@ufl.edu.
