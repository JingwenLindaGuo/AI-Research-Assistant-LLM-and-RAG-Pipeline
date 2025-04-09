# Research Assistant LLM Chatbot

## Overview

Our research assistant can help you get up to date with the latest research papers on the topics you like and want to know more about, the goal of the research assistant is to fetch papers based on the filters specified by the user then ingest the papers, and ansnwer the user queries based on the papers content.

## Deployment 
Our research assistant is deployed on huggingface you can find the link below:

[Research Assistant](https://huggingface.co/spaces/abdullah12100/LLM_assistant)

To use the model you need to create an account and login gain access to the chat interface.

## Clearml Evaluation
link to the model evaluation results: 
[Research Assistant](https://app.5ccsagap.er.kcl.ac.uk/projects/4581c1b73366408b92ac361df3110a7c/experiments/a58d7c69df8444c291f584f73153a336/info-output/metrics/plots?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-started&filter=)

## Local Deployment
1. Create a .env file with the following variables:
````
    OPENAI_API_KEY = "place your api key here"
    PINECONE_API_KEY = "place your api key here"
````
2. Create a python virtual environment:
````
    Python3.10 -m venv RAvenv
````
3. Install the requirements:
````
    pip install -r requirements.txt
````
4. Use this command to start the chatbot:
````
    Python app.py
````
5. Open a window on your browser and go to this address:
````
    127.0.0.1
````

##

Note: Commits made by 5aldx are from K21202785(Khaled albuainain), but they forgot to switch account
after using regular github for other courseworks.
