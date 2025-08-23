# LLMOps-and-AIOps-Skills

**(Cloud CI/CD Toolkit like Gitlab; Jenkins is a Local**

**A. AI Anime Recommender Project**

**E. Celebrity Detector and QnA**

**A. AI Anime Recommender Project**

1. Storage Base - CSV File containing Anime Name, Score, Genre, Synopsis; It is on basis of what Recommender will work (Anime - Japanese Art Style)

2. GCP VM - Virtual Machine that can be accessed on Cloud; It is a service offered by Google Cloud

3. Groq - LLM; We will use LLama3 LLM

4. HuggingFace - Embedding Model

5. LangChain - Generative AI Framework to interact with LLM

6. Minikube - For making a Kubernetes Cluster where we can deploy our application (K8S - Short form for Kubernetes)

7. Streamlit - To make UI or frontend of the application

8. Docker - For containerization of app during deployment

9. Grafana Cloud - For monitoring Kubernetes cluster (We can see how many namespaces are there, how many pods are running inside Kubernetes Clusters, How many nodes are running inside Kubernetes cluster and everything; All details with respect to Kubernetes will be shown in Grafana; We can set alerts, create dashboards)

10. ChromaDB - Local Vector Store for storing the embeddings

11. Kubectl - Command Line Interface to interact with Kubernetes (If we want to create a new pod, want to create a namespace, create a new node; We can use this Command Line Interface to do that)

12. GitHub - It will work as a Source Code Management (SCM) for our project

### We will convert CSV --> Documents --> Embeddings (Using HuggingFace Embeddings); And store in VectorStoreDB, ChromaDB

### In coding we will create two Pipelines - Training Pipeline and Recommendation (or Prediction) Pipeline 

### When we run trainging pipeline, it will extract CSV File, store it in ChromaDB

### Recommend Pipeline will accept user inputs and use it prompt templates and recommended the class to generate the output

### Docker file will also have on how we expose the Stremlit App (Say Port 8501)

### We will also do Kubernetes Injection - We have Groq API and HuggingFace API; We will push to GitHub for Versioning; But we can't push our environment files to GitHub because those are our Secret Keys; So we do Kubernetes Injection; We will Inject the API Keys directly into Kubernetes

### In GCP VM we will be installing three things: Docker Engine (Building Docker Image and Installing Minikube), Minikube, Kubectl (Need to interact with Kubernetes Cluster)

### We will integrate whatever we have in GitHub Code, and connect it to GCP VM; Once code has been copied we will have the docker file, Kubernetes file; We will build Docker Image using Docker file and Deploy in Kubernetes; After Github Integration, we will build Test App

**Project Codes:**

1. pip install -e . - Triggers setup.py (Install all the dependencies from setup.py)



**E. Celebrity Detector and QnA**

1. Groq - LLM (LLama 4 Vision Transformer)

2. Circle CI - CI/CD for making Pipelines **(Cloud CI/CD Toolkit like Gitlab; Jenkins is a Local**

3. OpenCV Python - To deal with Image work like conversion, scaling, etc..

4. Github - Serves as SCM for the project

5. Flask - To make the backend of the application **Helps user to upload Images**

6. HTML/CSS - To make UI or Frontend of the application

7. Docker - For containerization of app during deployment

8. GCP GAR - For storing Docker Images **Google Artifact Registry; It is like a Docker Hub for Google Cloud**

9. GCP GKE - To deploy and run app in Cloud within a Kubernetes Cluster; It is an service offered by Google Cloud **Google Kubernetes Engine**

**Project Working**

1. Created a folder, venv, requirements.txt,setup.py, .env,  folders- templates, static, app --> __init__.py, utils folder --> __init__.py

2. Run "pip install -e ." to run setup.py

3. Created image_handler.py inside utils using OpenCV **Creating Image Handler Code using OpenCV** - Used BytesIO, OpenCV, np

4. **Created Celebrity Detector Code using Llama-4** - Creating celebrity_detector.py in utils; **Used GROQ and Meta Llama-4**

5. **Created qaengine.py in utils - Question Answer Engine Code**

6. **Created Flask Backend Routes code using routes.py inside utils**

7. Writing the main code in "__init__.py" in Utils folder

### Here it will run in Flask in Local Host and Ports

8. Created a Docker File (Some of the OpenCV Library needs some dependencies)

9. Created Kubernetes Deployment yaml file

**We will inject API into Kubernetes Cluster**

10. Added .gitignore file for GitHub code versioning
