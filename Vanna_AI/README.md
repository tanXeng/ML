# Vanna AI

## Overview

Vanna.AI is an open-source AI-powered tool that simplifies database interactions by converting natural language questions into accurate SQL queries. Its value lies in democratizing data access, allowing both technical and non-technical users to extract insights from databases without extensive SQL knowledge, while also improving efficiency for experienced data analysts. By leveraging large language models and retrieval-augmented generation techniques, Vanna enhances data exploration, enables non-technical teams, and accelerates the process of obtaining actionable insights from complex databases.

This deployment of Vanna.AI is a single pod running the Vanna Flask UI with onboard ChromaDB as the vector store, and a [sample sqlite database.](https://github.com/lerocha/chinook-database)

## File Structure
```bash
Vanna_AI/
├── app/ # the folder containing the source code for the app
│ ├── assets
│ │ └── vanna_demo.gif # the app logo
│ ├── ai.py # AI pipeline for LLM SQL generation and database querying.
│ ├── app.py # not used in the flask app
│ ├── Chinook.sqlite # the sample SQLite database
│ ├── Dockerfile # the dockerfile used to build the docker image
│ ├── main.py # not used in the flask app
│ ├── models.py # Pydantic models and enums for FastAPI request/response schemas.
│ ├── onnx_download.py # not used in the flask app
│ ├── requirements.txt # requirements for the Dockerfile
│ ├── vanna_calls.py # not used in the flask app
│ └── vanna-flask.py # flask app entry point
├── vanna-ai-chart/ # the helm chart used to deploy the app
│ ├── templates
│ │ ├── ezua
│ │ ├── virtualService.yaml
│ │ ├── _helpers.tpl
│ │ ├── deployment.yaml
│ │ ├── service.yaml
│ │ └── virtualservice.yaml
│ ├── .helmignore
│ ├── Chart.yaml
│ └── values.yaml
├── README # this README file
└── .gitignore 
```
**Note:** some files in `Vanna_AI/app/` may not be used in the flask app.  

## How it works
![](https://vanna.ai/docs/img/how-vanna-works.gif)

1. *User Input:* The user submits a natural language query.
1. *Embedding Generation:* Vanna.ai generates embeddings for the user's query.
1. *Context Retrieval:* The system performs a vector similarity search to find relevant context from the trained data.
1. *LLM Processing:* The query and relevant context are sent to a Large Language Model (LLM).
1. *SQL Generation:* The LLM generates an SQL query based on the provided information.
1. *Query Execution:* Vanna.ai executes the generated SQL query against the connected database.
1. *Result Formatting:* The system formats the query results, typically as a Pandas DataFrame or Plotly figure.
1. *Response Delivery:* Vanna.ai returns the results to the user, often including the SQL query, data, visualizations, and potential follow-up questions.

# Deploying on HPE Private Cloud AI Cluster

The `vanna-ai-chart` Helm chart is designed to deploy a simple version of the Vanna AI application to HPE AI Essentials (Or other Kubernetes Clusters). The chart provides a flexible way to configure the application's settings and environment variables.

<!-- ## Prerequisites

* Access to a HPE Private Cloud AI Cluster running AI Essentials (AIE)
* An OpenRouter api token.
* A database you wish to connect to (it includes a sample sqlite Chinook DB)- currently supports SQLite and MS-SQL -->

## Configuration

The Vanna AI application requires several settings to be configured in the `values.yaml` file. These settings include:

| Setting | Description | 
| --- | --- |
| `Chatmodel` | The model on OpenRouter to use (any model that has sql in its training set should work). |
| `Chatmodelbaseurl` | The base URL for the OpenRouter API. |
| `Chatmodelkey` | The OpenRouter API key. |
| `Useinhousemodel` | If set to false, `Chatmodelkey` will be treated as the base URL for the OpenRouter API. Otherwise, `Chatmodelkey` will be treated as the endpoint of an LLM deployed on AIE using Machine Learning Inference Service. |
| `Database` | The database connection string. |
| `DatabasePath` | The path of where to persist the vector database. |
| `DatabaseType` | The type of database. |

**Note:**
You may need to configure other parts of the `values.yaml` file as per the requirements of your cluster. Do remember to change the `deployment.yaml` file under the `templates` folder in `vanna-ai-chart` to accommodate the changes in the `values.yaml` file.

## Packaging the Helm Chart
From the root of this folder, run:
`helm package vanna-ai`

