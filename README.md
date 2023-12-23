# final_project
This repository uses Graph4NLP library to evaluate Graph2Tree class on MAWPS dataset.
Graph to Tree interpretation of the math word equation is depicted as follows:
![Graph 2 Tree Interpretation](https://github.com/Anshita1Saxena/final_project/blob/main/images/mapws_graph_2_tree_interpretation.png)

Depiction from sentence to math word equation in the form of Graph input and Tree output:
![Graph 2 Tree](https://github.com/Anshita1Saxena/final_project/blob/main/images/g2t.png)

## Source Code Files:

1. Common functions such as accuracy computation and out-of-vocabulary functions are stored in `utils.py`.
2. The model implementation code is inside `Mawps.py`.
3. All hyperparameters are stored in `config.yaml` file. These hyperparameters include graph construction arguments (this graph is the embedding converted from natural language text data into graph embedding), graph initialization arguments, and decoder arguments.
4. The file which loads the `config.yaml` file into our project is available in `load_config.py`.
5. The file from which we run all the models experiments are kept inside `run_experiemnts.py`.

## Directory File Information:

1. data: We kept here two directories-
   a) Processed- This has the `NodeEmbGraph` directory which consists of the `data.pt` and `vocab.pt` files which are used for creating the graph embeddings.
   b) Raw- This directory holds the raw MAWPS data where we split the dataset into 80% train, 10% validation, and 10% test dataset.
2. checkpoint_save: This directory holds all the files that have the history of some of our experiments. We did not upload all the run history files here as this is not the best practice. We uploaded some files for sample runs.
3. experiements: We initiated to run the model and experiments with Jupyter Notebooks first and then we came up with this modular structured approach to formally ran the model and experiments.

## How to run this project:

Libraries to install:-
1. Graph4Nlp- pip install graph4nlp
2. Torchtext- pip install torchtext
3. Torch- pip install torch
4. Numpy- pip install numpy

Tools:- CoreNLP
This tools is published by Standford to work with NLP processing tasks. This is available as JAR file, on Huggingface, and as Maven. For this project, we used JAR file to properly setup this tool. We need to navigate to this website: [CoreNLP Software Link](https://stanfordnlp.github.io/CoreNLP/) This software is used in building the graph embeddings. From this page, we downloaded the JAR files (binaries) to our system. On command prompt we need to provide the following command to start the server:
`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 9001 --timeout 15000`
Reference figure is shown here:-
![CoreNLP Connection Screenshot](https://github.com/Anshita1Saxena/final_project/blob/main/images/corenlp_server_start.png)

As stated above, there are two ways to run this project:- 
1. Using the Modular Approach: You can first clone this repository into your system. Start the core nlp server and run the experiments using the command on terminal:
`python run_experiments.py`

2. You can use the jupyter notebook ipynb file stored in the `experiment` directory and run each cell sequentially after connecting to the corenlp server.

## Model Architecture for Project Flow:
The model architecture is depicted as below. Entire report for this project is uploaded on GradeScope for the Assessment:
![Model Flow Architecture](https://github.com/Anshita1Saxena/final_project/blob/main/images/graph4nlp_flow.png)
