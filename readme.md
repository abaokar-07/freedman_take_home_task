Setup instructions:

1. Clone this repo into your local machine
2. Run the file setup.py to install all the required libraries. Use the command "python setup.py install"
3. The module 'torch', 'faiss-cpu' and 'numpy==1.26.0' will need to be installed separately as
    a. torch has platform specific wheels
    b. faiss-cpu needs numpy version 1.26.0 ro run
3. Open the file "config_qa.py". In this file, replace your Hugging Face token and also set the folder paths for the cache so that after the first time, model will load from the cache
4. Run the file "main.py". The terminal will generate a local link for the gradio app where the question answering UI will be present.