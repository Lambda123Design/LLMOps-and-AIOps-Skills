# LLMOps-and-AIOps-Skills

**To Trigger setup.py - pip install -e .**

**(Cloud CI/CD Toolkit like Gitlab; Jenkins is a Local**

### **We are converting here; Why not RGB to Gray; Because when we get a Image from Internet it is BGR and not RGB** gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### Next is temperature; Higher the temperature, more creative our model is; More creative more hallucinative, if suppose we have given Virat Kohli, it can detect as other clebrity, so give it in between 0.2-0.5

#### "Full Documentation.md" file of tutor in GitHub has every steps that we have to do for Deployment

**A. AI Anime Recommender Project**

**B) Flipkart Product Recommender using Prometheus,Grafana,Minikube,AstraDB, LangChain (Will do Later)**

**E. Celebrity Detector and QnA**

**F. AI Music Composer**

**G) Multi AI Agent using,Jenkins,SonarQube,FastAPI,Langchain,Langgraph,AWS ECS (Will do Later)**

**A. AI Anime Recommender Project**

**1. Introduction to the Project:**

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

**2. Project and API Setup ( Groq and HuggingFace )**

Hello everyone. So this is the project setup video of this project. In this video basically we will be doing our project structure, then our API setup, and everything else like logging, custom exception, and virtual environment formation. Okay. So first of all just come to any of your folder and let’s create one new folder here and let’s name it okay. Let’s name it. Since we are making an anime recommender system, let’s name it Anime recommender okay, like this. Now open your anime recommender. Just right click on the folder and open it in terminal and you can just write "code ." (make sure it is in small letters "code full stop"). So basically it will trigger your VS Code automatically. Let’s click it. You can see our VS Code has been opened automatically. Let’s give the permission.

Now open the terminal on the top, you will get new terminal. Let me maximize the terminal first of all and make sure you open the terminal in command prompt. Okay, let me increase the size so that you can see correctly. First of all, let’s create a virtual environment. So I will write "python -m venv venv" and whatever you want to name your virtual environment. So let’s name it "venv" only. Okay, press enter. On the left pane you can see a venv has been created. So after it has been created, we will activate it. Okay, let’s wait. Why is it taking too much time? Okay, it has been created. Now we have to activate. First of all, give the virtual environment name, we have given "venv", then you have to give the backslash, then "Scripts", then again a backslash and "activate". So it will basically activate your virtual environment. Okay okay I have written the wrong spelling for Scripts, it is "Scripts". Ah I press enter. You can see in the bracket you are getting "venv". It means your virtual environment has been activated. So let me clear the screen with "cls".

Okay now let’s do our API setup. Basically we will be needing two APIs for this project. So first of all in the root directory let’s create one .env file. So I will create one environment file. So what are these two APIs? First of all, we will be needing our Grok API key, "GROK_API_KEY". Then, in between these quotes we will be placing our Grok API key. Another thing we need is Hugging Face API, so I will write "HUGGINGFACEHUB_API_TOKEN". Make sure the spelling is correct. And most importantly, this one, this Hugging Face, it should be exactly what I have done: "HUGGINGFACEHUB_API_TOKEN". Like this.

Now where will you get these APIs? Just go to your browser. Open a new tab and search for Grok Cloud API. Just make an account on Grok Cloud, and just go to the API key section, create an API key, and give any name. Since we are making an anime recommender, let’s name it "anime" only. Submit it and copy it, and in the VS Code, paste it here. Done. Now coming to Hugging Face, let’s search for Hugging Face. You have to do the same, you have to make an account on Hugging Face first of all. When you make your account, then you have to sign in into your particular account. Then click on the profile section and you will get access tokens. Just click on access tokens. Basically it will ask your password for identity confirmation, and you have to create a new token. Make sure it is in write mode. There are three modes: fine grain, read, and write. It should be in write mode. Give the token name "anime" only and create the token. You are getting the token, just copy this token again and come to VS Code and paste it here. Okay then.

So these were your two APIs that you need. You have set up your virtual environment also. Now let’s come to our basic project structure. Our project will look like this. First of all, we will be needing some data. We will be needing some data on which we will make our vector store and everything. Our recommender will work on something, right? So we will be working on an anime data. So let’s create one "data" folder here. From where will you get the data? Just come to the resource section. In the resource section you will get one GitHub link. In the GitHub link you will get one data folder here same, and you will get this "anime_with_synopsis.csv". I will provide it in the resource section directly. Also, make sure you open this "anime_with_synopsis.csv" so it will look like this. Basically these are your MAL IDs (MAL means MyAnimeList, a very popular anime website). So these are your anime names, then these are your genres, and these are your synopsis. On the basis of this genre and the synopsis, we will be recommending anime.

For example, let’s say Naruto. The most similar anime related to Naruto is Black Clover. And we can also say Bleach. Bleach is also an anime, and the most similar anime to Bleach is, I think, Jujutsu Kaisen. Right? So let’s download this CSV file. Now let me open this download folder, and what you can do is just minimize it, open your downloads, and copy it from here inside your data directory. Okay, now your data is inside your data directory.

Then, now let’s make our project structure also. First of all let’s make one "requirements.txt". Basically this file will contain all your requirements, all your libraries that you need during your project. So make sure you name it "requirements.txt". Okay. Now what are the libraries that you need? First of all it’s "langchain", we will be using LangChain. Then we will be using "langchain_community". Then we will be using "langchain_groq", basically we are interacting with our Grok API, right, so we need a particular library for that. It is basically a modified version of LangChain only that is particularly made to handle Grok API key. Then we need "chromadb". So basically this will be used to create our vector store. In one of the projects we used FAISS vector store. In this project we are using ChromaDB. Okay. Then we need "streamlit". We need Streamlit for making the UI part. Then we need "pandas" also. Basically this is a CSV file, right? So we can’t directly connect with our CSV files. We have to use pandas, convert it into dataframe, then okay. After that we need "python-dotenv". Basically this library will be used to load our environment variables. You can see we have two environment variables, so how can you load them inside your project? By using python-dotenv. Then we need "sentence-transformers". After sentence-transformers we need LangChain but in LangChain we need "langchain_huggingface". Basically you have two APIs: Grok and Hugging Face. For interacting with Grok you will be needing "langchain_groq", and for interacting with Hugging Face you need "langchain_huggingface". Clear? So these are your requirements, all the requirements.

Now you will be creating one "setup.py". What’s the use of setup.py file? To install all your requirements, install all your dependencies, install all your packages. Right. So I have already explained the code for setup.py, so I will be directly writing the code. You can just go to the GitHub link that I have given, you will see their setup.py file. Just copy it and paste it here. Okay, let’s change the name of the setup to something like "Anime Recommender". Like this. Okay, now how can you trigger your setup.py? For that you will write "pip install -e ." in your terminal. So basically it will automatically trigger your setup.py file. Press enter. So it has started building all the dependencies. It will install all the requirements. Also, currently we don’t have any packages so it will not install packages okay. But we will make packages also, and then again we will run the same command "pip install -e ." again and again whenever we will create one package. Right. Let’s keep it running. Let me cancel the setup.py right now.

Now let’s make our project structure for now. Come to the root directory only and create one "src" folder here. Basically inside the src folder your main components will be there, like your data loader, your prompt templates, your recommender, your vector store, like that. And since src will be treated as a package, you have to create one "__init__.py" inside it. So this was all about your src package. Now you have to create another folder that is "utils". Utils basically means utilities. Inside that you can create your helper functions, your logger function, and your custom exception functions. Since we also want to import functions from utils folder, we have to make it as a package. For making a package you will create "__init__.py" and inside utils you will create two more files like "logger.py" and "custom_exception.py". I have already explained these two files in the starting of the course, so I will directly copy-pasting them to the utils. Go to the logger.py, just copy it, come to the logger.py and paste the code here. Okay. Come to the custom exception, do the same, just copy paste the code. Done. Let me close all the files. So these are your utils. These are basically your utilities. If suppose moving forward we want to create some helper functions also, we will create them inside utils only. You can just name another file "helpers.py", and inside that you can create your helper functions.

Then let’s create one another folder. Let’s name it "config". Inside config you will be giving all your configurations basically like your API setups, your Hugging Face setup, and suppose if you are using a database, you can do the database setup also inside this configuration folder. And since configuration folder will also be treated as a package, make sure you convert it into "__init__.py". Then okay. Now we will create one "app" folder also, and let’s keep it also as a package. Basically inside this folder we will be creating our main application, mainly our Streamlit application. Then we have another folder also, let’s make it "pipeline". Basically here we will be making our whole pipeline. Basically, in the src directory you will be creating all the components, and in the pipeline directory you will be joining all the components to form a pipeline. And since pipeline will also be treated as a package, make sure you convert it into "__init__.py".

I think this was it. I think we need app, we need configuration, we created it. We already have data folder, we have pipeline, we have src, we have utils. Yeah, everything is done. So this was all your setup. This app folder will be required for creating your application, config for configuration, data has data inside it, pipeline will be used to join all the components that are inside this src folder. In utils you will create your helper function, your logger function, custom exception function like that. This is your .env which contains two APIs, then your requirements, then setup.py. Currently your requirements are getting installed. Once these requirements get installed, you have to again run that "pip install -e ." command. Why? Because we have created packages. This app is a package. This config is a package. This pipeline is a package. Whenever you will get an init file, that folder is a package. This data directory is a folder, but this pipeline directory is a package. Understand? Then src is also a package. Then utils is also a package. So you have to again run that "pip install -e ." so that all these folders will be created as packages from now on. Okay. So I will not be doing it here. Once these are all installed and once you again run the pip install -e command, then you can move to our next video.

**Summary:**

The first step in setting up the Anime Recommender System project is to create a dedicated folder named Anime Recommender. This folder is opened in VS Code using the command code .. A virtual environment is then created inside this folder using python -m venv venv and activated with venv\Scripts\activate. This ensures that all the project dependencies are isolated and managed properly.

Next, the project requires API setup. A .env file is created in the root directory to store API keys securely. Two APIs are needed: the Grok API and the Hugging Face API. Users must create accounts on their respective platforms, generate API keys or tokens, and paste them into the .env file under GROK_API_KEY and HUGGINGFACEHUB_API_TOKEN. The Hugging Face token should have write permissions.

The project’s data is organized in a data folder. The main dataset, anime_with_synopsis.csv, contains anime IDs, names, genres, and synopses. This dataset will be used to build the vector store for recommendations. Users download the CSV from a provided GitHub link and place it in the data folder.

The project structure also includes several key folders and files. A requirements.txt lists all necessary Python libraries, such as langchain, langchain_community, langchain_groq, chromadb, streamlit, pandas, python-dotenv, sentence-transformers, and langchain_huggingface. A setup.py file is used to install these dependencies and register the project packages. The main folders include src for core components like data loaders, prompt templates, and vector stores, pipeline to connect components into a complete workflow, utils for helper functions, logging, and custom exceptions, config for API and database configurations, and app for the Streamlit application. Each folder that contains Python code is converted into a package with an __init__.py file.

Finally, after setting up the folders, APIs, and requirements, the command pip install -e . is run to install all dependencies and register the packages. This command needs to be re-run whenever new packages are added. With this setup complete, the project is ready for further development, including building the recommender logic, connecting the APIs, and creating the UI.

**3. Configuration Code**

Hello everyone. So in this video we will be doing our configurations, right? But before going to the configuration, I already told you that you have to again run "pip install -e ." so that it will install all the packages. Also, we have made our app as a package, config as a package, pipeline as a package, src as a package, and utils as a package also. So you have to again run this "pip install -e ." and it will generally run fast. Fast. Okay.

So till it’s getting all the packages installed, just come to your config directory and you will create one file here that is "config/config.py". And inside it we will be creating our configurations. Okay. You can see it has started installing, okay successfully built and done. Okay. Now you can close the terminal.

Now for configuration we have to import some things. First of all, we have to import OS library. Then we have to import our dotenv by writing "from dotenv import load_dotenv". And we have to load it like this: "load_dotenv()". Now what this "load_dotenv" is doing is basically loading all our environment variables. You can see we have two environment variables. Now these two environment variables have been successfully loaded inside our project using this command "load_dotenv()".

Now let’s create one variable here. Let’s name our variable as "API_KEY". Currently, this Grok API key is not a Python variable. It’s an environment variable. We have to convert it into a Python variable, so we are doing that only. So this "API_KEY" and the Grok API key inside the .env are different. This is an environment variable, and inside this config, this is a normal variable. So I will write "API_KEY = os.getenv('GROK_API_KEY')". Now what this line is basically doing is fetching that particular environment variable "GROK_API_KEY" and storing its value inside this simple Python variable. Okay, now our API key is stored inside this normal variable.

Now another thing is "MODEL_NAME". Basically Grok offers us a wide range of models. I can show you also. Just come to the browser and write "Grok Cloud models" like this. Just go on the first site "Supported Models". And you can see these are your models which are supported. These are production models, and then there are preview models also. Let’s talk about the production models right now. So basically, these two are widely used: "3.3 70B Versatile" and "3.18B Instant". So let’s use a normal model only. Let’s use a small model. So we will be using "8,000,000,001". Okay, let’s copy its name, come to your VS Code and paste that name here like this: "MODEL_NAME = '8,000,000,001'".

So this was basically all your configurations. Basically, we configured our API key, and we configured our model name only. These are the two things that you need to configure. Okay. Now let’s move to our next video.

**Summary:**

Before starting the configuration, it’s important to run pip install -e . again to ensure all packages and project folders (app, config, pipeline, src, utils) are registered and installed properly. Once the packages are installed, you can proceed to create the configuration file inside the config directory as config/config.py.

In this file, the OS library is imported along with load_dotenv from the dotenv library. By running load_dotenv(), all environment variables stored in the .env file (like the Grok API key and Hugging Face token) are loaded into the project.

Next, the environment variables are converted into Python variables for easier use in the project. For example, the Grok API key from the .env file is assigned to a variable using API_KEY = os.getenv('GROK_API_KEY'). Similarly, the model to be used from Grok Cloud is configured by setting a variable MODEL_NAME to one of the supported models, such as '8,000,000,001'.

In summary, the configuration step involves loading environment variables and defining Python variables for the API key and the Grok model name. With this configuration in place, the project is ready to start integrating the APIs and building the recommender system.

**4. Data Loader Class Code**

Hello everyone. So in this video we will be doing our code for the data loader. So come to your "src" folder, and inside "src", create one file here that is "data_loader.py". Now let's write the code for this data loader. First of all, we need pandas since we are dealing with our CSV files. Okay, "import pandas as pd". Like this.

Let's create one class here. Let's name the class as "AnimeDataLoader". Okay, that's my anime data loader. Like this. Let's make the constructor "__init__". First of all, there will be a "self" parameter. Then we will give our original CSV and make sure the path is in string, and then we will be giving our processed CSV, and it should also be string. Okay. So basically this class requires two things: you have to pass the path of the original CSV, and you have to give the path for the processed CSV. Basically, we will be using this original CSV that is "anime_with_synopsis.csv", and after doing some changes with this CSV, we will be making one new CSV.

Because, as you can see, this "anime_with_synopsis.csv" has some unnecessary columns. Also, you can see this "MAL_ID"—we don't require it. This "score"—we don't require it either. You can see these are unnecessary columns. We can also combine the "genre" and "synopsis" into one column only. We can do this as a form of data construction, feature construction, as you have learned in machine learning. Right?

So I will write some instance variables: "self.original_csv = original_csv" and "self.processed_csv = processed_csv". Okay, so this was your constructor. Now let's make our first method, that is "load_and_process". "def load_and_process(self):". Now what you will be basically doing: first of all, let's read the CSV. Basically, you can't make changes to a CSV file directly. You have to convert it into a pandas DataFrame. So I will write "df = pd.read_csv(self.original_csv, encoding='utf-8', error_bad_lines=False)".

Now what this line is doing: first, we are reading the original CSV, then we are specifying the encoding as "utf-8" because sometimes a CSV can be in some other encoding, like Latin encoding. But we want UTF-8 because UTF-8 is a standard encoding. Then "error_bad_lines=False" will skip any bad lines. Bad lines are lines where you might get a deprecation warning or malformed rows. After that, we will do "df = df.dropna()" to drop all rows that have null values so that our data is clean.

Now let's give a variable "required_columns" because we only require some columns. I will write "required_columns = ['name', 'genre', 'synopsis']". Make sure you copy the names properly; there should be no mistakes in the column names. Now we can check if any column is missing: "missing = set(required_columns) - set(df.columns)". If "missing", then raise a value error: "raise ValueError('Missing column in CSV file')". What this is doing is: first, we set up the required columns. Then we check if the DataFrame contains all of them. If any required column is missing, the "missing" set will have some value, and the error will be raised. If all columns are present, the set "missing" will be empty and this block will be skipped.

Now let's create a new column. We will create "df['combined_info'] = df['name'] + ' ' + df['overview'] + ' ' + df['synopsis'] + ' ' + df['genre']". What is happening here: we are combining the three columns "name", "genre", and "synopsis" into a single column called "combined_info". For example, the first row "Cowboy Bebop" will have its title, then the overview, then synopsis, then genre all combined into one string in the "combined_info" column. This is how our final combined column will look.

Now we will create a new CSV file that will only consist of this combined info. We will remove the original "name", "genre", and "synopsis" columns, and save a new CSV which only has "combined_info". So I will write "df[['combined_info']].to_csv(self.processed_csv, index=False, encoding='utf-8')". Finally, we return "self.processed_csv".

So this was all your "AnimeDataLoader" class. To summarize: first, we made our data loader class which accepts two parameters: the original CSV and the processed CSV path. We read the original CSV, selected required columns, performed a check for missing columns, dropped null rows, combined "name", "genre", and "synopsis" into a single column, exported the DataFrame with only "combined_info" to a CSV with "index=False" and "encoding='utf-8'", and returned the processed CSV path.

This was all your "AnimeDataLoader" class. Now let's move to our next video.

**Summary:**

In this video, the focus is on creating a data loader for the Anime Recommender System. Inside the src folder, a file named data_loader.py is created, and a class AnimeDataLoader is implemented. This class requires two parameters: the path to the original CSV (anime_with_synopsis.csv) and the path for the processed CSV. The purpose of this class is to read the original CSV, clean it, and prepare a processed version for the recommender system.

The load_and_process method reads the CSV into a pandas DataFrame using UTF-8 encoding and skips any malformed lines. It then drops rows with null values to ensure data quality. Only the necessary columns (name, genre, synopsis) are selected, and a check is performed to raise an error if any required column is missing. To prepare the data for recommendation, the method combines the name, genre, and synopsis columns into a new column called combined_info. This combined information is then saved into a new CSV containing only the combined_info column, which is returned as the output of the method.

In summary, the AnimeDataLoader class efficiently handles CSV reading, cleaning, validation, and feature construction by combining relevant columns into a single processed dataset, making it ready for building the recommendation system.

**5. Vector Store Code using Chroma**

So in this video we will be dealing with our vector store. First, come to your "src" folder and create a file called "vector_store.py". Here we will write the logic for our vector store. First, we need to do some imports. From LangChain, we import the text splitter using "from langchain.text_splitter import CharacterTextSplitter". This splitter will help separate our text; since we combined "name", "genre", and "synopsis" in the previous step, the splitter will separate text like "action, adventure, comedy, drama" and the whole synopsis as well.

Next, we need the vector store itself. For this, we import Chroma using "from langchain.vectorstores import Chroma". We also need to import the CSV loader because the processed CSV saved in the previous video must be loaded to convert into a vector store, using "from langchain.document_loaders import CSVLoader".

To convert the text into embeddings, we use Hugging Face embeddings. We import it with "from langchain.embeddings import HuggingFaceEmbeddings". Hugging Face embeddings automatically fetch your API token from environment variables. To load these environment variables, we write "from dotenv import load_dotenv" and then "load_dotenv()".

Now, let's create a class for our vector store just like we did for the data loader. We define the class with "class VectorStoreBuilder:" and a constructor "def init(self, csv_path: str, persist_directory: str):". Inside the constructor, we define three instance variables: "self.csv_path = csv_path", "self.persist_directory = persist_directory", and "self.embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')". Here, the embedding model "all-MiniLM-L6-v2" is from Hugging Face sentence transformers, which we installed earlier using "sentence-transformers".

Next, we create a method to build and save the vector store, defined as "def build_and_save_vectorstore(self):". First, we load the CSV using "loader = CSVLoader(file_path=self.csv_path, encoding='utf-8', metadata_columns=[])" and "data = loader.load()". Then we split the loaded data using the character text splitter: "splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)" and "texts = splitter.split_documents(data)". After splitting, we save the text into the vector store with embeddings using "db = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=self.persist_directory)" and persist it locally with "db.persist()".

We also create a method to load an existing vector store if we already have one. This method is defined as "def load_vectorstore(self):" and simply returns "Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)".

In summary, first we import necessary modules, then define the class "VectorStoreBuilder" with CSV path, persist directory, and Hugging Face embeddings as instance variables. The "build_and_save_vectorstore" method loads CSV data, splits it into chunks, converts chunks into embeddings, and saves them locally in Chroma. The "load_vectorstore" method allows loading the existing vector store directly without rebuilding.

This is how we create and manage the vector store for our anime recommender system, and this sets the stage for building the pipeline in the next step.

**Summary:**

In this video, the focus is on creating a vector store for the Anime Recommender System. Inside the src folder, a file named vector_store.py is created, and a class VectorStoreBuilder is implemented. This class is responsible for converting processed text data into embeddings and storing them in a persistent vector database using Chroma. Required imports include CharacterTextSplitter for splitting text, Chroma for the vector store, CSVLoader for loading the processed CSV, and HuggingFaceEmbeddings to generate embeddings from text. Environment variables are loaded using dotenv to access the Hugging Face API token.

The class constructor takes the CSV path and a persist directory as input and initializes the Hugging Face embedding model (all-MiniLM-L6-v2). The build_and_save_vectorstore method first loads the processed CSV, splits the combined text into chunks using the character text splitter, converts the chunks into embeddings, and saves them in the Chroma vector store. A separate method, load_vectorstore, allows loading an existing vector store directly without rebuilding, which saves time for subsequent runs.

In summary, the VectorStoreBuilder class handles loading processed CSV data, text splitting, embedding generation using Hugging Face, and persistent storage with Chroma, forming the backbone for similarity search in the anime recommender system. This setup prepares the project for building the recommendation pipeline in the next steps.

**6. Prompt Templates Code**

So in this video we will be creating our prompt template. First, go to your "src" directory and create a file named "prompt_template.py". I won’t write the full prompt template file here because it is very long and tedious. I have generated it using ChatGPT as well, so you can just go to the GitHub directory mentioned in the resource section, open "prompt_template.py", and copy the full file. I will explain it here.

First of all, we do an import with "from langchain.prompts import PromptTemplate". Prompt templates are basically instructions to the LLM (Large Language Model) telling it how to perform a particular task. In our case, we are making an anime recommender system, so the prompt tells the LLM to act as an expert anime recommender.

We define a function called "get_prompt()". Inside it, we define the template text: "You are an expert anime recommender. Your job is to help users find the perfect anime based on their preferences. Whatever the user wants, like if they want a similar anime to Attack on Titan, your job is to give a perfect recommendation that is similar to Attack on Titan. Use the following context and provide a detailed and engaging response. For each recommendation, give the anime title, a 2–3 sentence plot summary, and a clear explanation why this anime matches the user’s input. Present your recommendations in a numbered list. If you don’t know the answer, respond honestly by saying that you don’t know. Do not fabricate answers."

Next, we define the input variables for the template. We have two input variables: "context" and "question". Here, "context" is the data source (from our vector store), and "question" is the user’s query, like "Give me suggestions similar to Naruto". Finally, we return the prompt template object using "PromptTemplate(template=template, input_variables=['context','question'])".

This setup ensures that the LLM receives both the context (from the vector store) and the user’s question and outputs a structured, accurate, and engaging anime recommendation. You can also modify the template to improve its instructions or make it more descriptive using ChatGPT or other tools. You can even write it manually if needed.

That’s it for the prompt template. Once this is ready, the next video will cover an important part of the pipeline, which builds on this prompt template for actual recommendations.

**Summary:**

In this video, the focus is on creating a prompt template for the Anime Recommender System. Inside the src folder, a file named prompt_template.py is created. The prompt template serves as the instruction set for the Large Language Model (LLM), guiding it to behave as an expert anime recommender. The langchain.prompts.PromptTemplate module is used to define the template.

A function get_prompt() is created, which contains the template text instructing the LLM to provide anime recommendations based on user preferences. The prompt specifies that recommendations should include the anime title, a short plot summary, and a clear explanation of why the anime matches the user’s query. Input variables for the template include context (data from the vector store) and question (the user’s query, e.g., “Give me suggestions similar to Naruto”). The function returns a PromptTemplate object that combines both the context and the user query to generate accurate, structured, and engaging recommendations.

In summary, the prompt template ensures the LLM receives both the relevant data and user input, producing meaningful anime recommendations. This template forms a key part of the recommendation pipeline, guiding how the LLM interprets and responds to queries.

**7. Recommender Class Code**

In this video we will be making our anime recommender class. First, go to your "src" directory and create a file named "recommender.py". Then we start with the imports. We need "from langchain.chains import RetrievalQA" to create the retrieval question-answer chain. We also need "from langchain.chat_models import ChatGrok" to interact with our LLM using ChatGrok. Finally, we import our prompt template with "from src.prompt_template import get_prompt" which we created in the previous video.

Next, we create a class called "AnimeRecommender". First, we define the constructor with "def init(self, retriever, api_key: str, model_name: str)". These are the inputs: a retriever, the API key, and the model name, which we already have in our configuration.

Inside the constructor, we first initialize our LLM using ChatGrok: "self.lm = ChatGrok(api_key=api_key, model=model_name, temperature=0)". Here, temperature is set to 0 to restrict the LLM’s creativity. This ensures it provides clear and factual recommendations without making up answers if it doesn’t know something.

Next, we load the prompt template using "self.prompt = get_prompt()". Then we create the retrieval question-answer chain: "self.qa_chain = RetrievalQA.from_chain_type(llm=self.lm, chain_type='stuff', retriever=retriever, return_source_documents=True, chain_type_kwargs={'prompt': self.prompt})". Here, the chain type "stuff" ensures all documents in the retriever are pulled as context for the prompt, and "return_source_documents=True" keeps track of the documents used.

Next, we define the recommendation method: "def get_recommendation(self, query)". Here, "query" is the user’s question. We get the recommendation using "result = self.qa_chain.run(query)". The result is a dictionary containing multiple keys, but we only return the final recommendation using "return result['result']".

To summarize: we imported ChatGrok, RetrievalQA, and our prompt template. We initialized the LLM with API key, model name, and temperature. We loaded the prompt template and created the QA chain using the retriever and chain type. Finally, we defined the "get_recommendation" method which runs the user query through the QA chain and returns only the recommendation.

This completes the Anime Recommender class. In the next video, we will integrate all the modules together into a working anime recommendation pipeline.

**Summary:**

In this video, the focus is on creating the AnimeRecommender class inside the src/recommender.py file. The class integrates the previously built modules—vector store, prompt template, and LLM—into a working recommendation engine. Required imports include RetrievalQA from LangChain for building a retrieval-based question-answer chain, ChatGrok to interact with the LLM, and the get_prompt function from the prompt template.

The AnimeRecommender class constructor takes a retriever, API key, and model name as inputs. It initializes the ChatGrok LLM with the API key and model name, setting temperature=0 to ensure factual and consistent responses. The prompt template is loaded, and a RetrievalQA chain is created using the retriever. This chain uses the “stuff” chain type to provide all relevant documents as context and keeps track of the source documents used.

The class includes a get_recommendation(query) method that accepts a user query, runs it through the QA chain, and returns the generated anime recommendation. In summary, this class brings together the LLM, prompt template, and vector store retriever to deliver structured, accurate, and context-aware anime recommendations, forming the core of the recommendation pipeline.

**8. Training and Recommendation Pipeline**

In this video, we will be creating our pipeline. Now, what is a pipeline? We have already created the data loader, prompt template, recommender, and vector store. The pipeline combines all these source files in a sequential way so that we can run just the pipeline and all components get processed in order. First, go to the "pipeline" directory and create a file named "pipeline.py".

We start with the imports. First, we import the vector store builder using "from src.vector_store import VectorStoreBuilder". Next, we import our anime recommender using "from src.recommender import AnimeRecommender". From the configuration file, we import the API key and model name using "from config.config import API_KEY, MODEL_NAME". We also import logging and custom exceptions using "from utils.logger import get_logger" and "from utils.custom_exception import CustomException".

Next, we initialize our logger using "logger = get_logger('magic_method')".

Now we create a class for the pipeline called "AnimeRecommendationPipeline". In its constructor "def init(self, persist_directory='chroma_db')", we use a try-except block to handle exceptions. First, we log "logger.info('Initializing recommendation pipeline')". Then, we create the vector store builder object using "vector_builder = VectorStoreBuilder(csv_path='', persist_directory=persist_directory)". Here, the CSV path is empty because we only want to load an existing vector store, not create a new one.

Next, we load the retriever using "retriever = vector_builder.load_vector_store().as_retriever()". This retriever is passed to the anime recommender class: "self.recommender = AnimeRecommender(retriever=retriever, api_key=API_KEY, model_name=MODEL_NAME)". After that, we log "logger.info('Pipeline initialized successfully')". In case of exceptions, we catch them with "except Exception as e" and log "logger.error(f'Failed to initialize pipeline: {str(e)}')" and optionally raise "CustomException('Error during pipeline initialization')".

Next, we define the recommendation method "def recommend(self, query: str) -> str". Inside a try-except block, we log the received query using "logger.info(f'Received query: {query}')". Then we call the recommender method to get the recommendation: "recommendation = self.recommender.get_recommendation(query)". Once the recommendation is generated, we log "logger.info('Recommendation generated successfully')" and return the recommendation using "return recommendation". In case of exceptions, we log "logger.error(f'Failed to get recommendation: {str(e)}')" and raise "CustomException('Error during getting recommendation')".

Now let’s discuss the build pipeline. This is in "build_pipeline.py". We import the data loader using "from src.data_loader import AnimeDataLoader" and the vector store builder again using "from src.vector_store import VectorStoreBuilder". We also load environment variables using "from dotenv import load_dotenv" and initialize the logger similarly.

Inside a main function "def main()", we use a try-except block. First, we log "logger.info('Starting to build pipeline')". Then, we create the data loader object using "loader = AnimeDataLoader(original_csv='data/anime_with_synopsis.csv', process_csv='data/anime_updated.csv')" and call "processed_csv = loader.load_and_process()". After that, we create the vector store builder object using "vector_builder = VectorStoreBuilder(csv_path=processed_csv)" and build the vector store using "vector_builder.build_and_save_vector_store()". We log "logger.info('Vector store built successfully')" and "logger.info('Pipeline build successfully')". In case of exceptions, we log the error and raise a custom exception.

The difference between these two pipelines is important. The "build_pipeline.py" is our training pipeline—it loads the data using the data loader and creates the vector store. The "pipeline.py" is our prediction pipeline—it uses the existing vector store to generate recommendations during runtime.

Finally, to execute the build pipeline, run "if name == 'main': main()". This ensures that the main method is executed when running "python pipeline/build_pipeline.py". After fixing any CSV read errors (e.g., using "on_bad_lines='skip'"), the logs will show "Data loaded and processed" and "Vector store built successfully", confirming that the pipeline is working.

Now you have your vector store ready. In the next video, we will create our main application which will use this "pipeline.py" for recommendations.

**Summary:**

In this video, the focus is on creating the pipeline for the Anime Recommender System, which integrates all previously built components—data loader, vector store, prompt template, and recommender—into a sequential workflow. A file named pipeline.py is created in the pipeline directory, and the AnimeRecommendationPipeline class is implemented. Required imports include the VectorStoreBuilder, AnimeRecommender, API key and model name from the configuration, as well as logging and custom exception utilities.

The pipeline class constructor initializes the pipeline by loading an existing vector store and creating a retriever. This retriever is then passed to the AnimeRecommender class along with the API key and model name. Logging is used to track the initialization process, and any exceptions are caught and handled using a custom exception. The pipeline also includes a recommend(query) method, which logs the user query, runs it through the recommender, and returns the generated recommendation while handling potential errors.

Additionally, a build pipeline (build_pipeline.py) is created for training purposes. This pipeline uses the AnimeDataLoader to process the original CSV, then builds and saves the vector store using VectorStoreBuilder. Logging ensures that each step is tracked, confirming successful data loading and vector store creation. The key distinction is that the build pipeline is used for creating the vector store (training phase), whereas pipeline.py is used for generating recommendations in runtime (prediction phase).

In summary, the pipeline consolidates all components into a structured workflow, enabling seamless data processing, vector store handling, and recommendation generation. Once the vector store is ready, the system is prepared for integration into the main application.

**9. Main Application Code**

In this video, we will be creating our main application. First, go to your "app" directory and create a file named "app.py". This file will contain all the logic for running our Streamlit-based anime recommender application.

We start by importing Streamlit using "import streamlit as st". Next, we import our prediction pipeline using "from pipeline.pipeline import AnimeRecommendationPipeline". We also load environment variables using "from dotenv import load_dotenv".

Let me briefly explain the recommendation pipeline again. The pipeline initializes the vector store that we have already created and loads it as a retriever. The retriever fetches relevant documents when a user query is passed—for example, if someone asks for recommendations similar to Naruto, it fetches related anime documents. Then, the language model uses these retrieved documents along with the prompts to generate recommendations.

We import the pipeline class "AnimeRecommendationPipeline" and call "load_dotenv()" to ensure all environment variables are loaded. Although we already loaded environment variables in the config file, we do it here again because, when deploying via Docker on platforms like AWS or GCP, the app.py file is the entry point. Loading environment variables ensures the API key and Hugging Face hub token are accessible.

Next, we set the Streamlit page configuration using "st.set_page_config(page_title='Anime Recommender', layout='wide')". This step is mandatory for a proper Streamlit UI.

To initialize the recommendation pipeline efficiently, we define a function with caching:

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()


This ensures the pipeline is initialized only once at app start, saving time for repeated queries. We then call "pipeline = init_pipeline()" to have a ready-to-use pipeline instance.

Next, we set the title of the app using "st.title('Anime Recommender')". We ask the user for input using "query = st.text_input('Enter your anime preferences', 'A light-hearted anime with school setting')". This provides a text input box for the user to type their anime preference or query.

Once the user enters a query, we use a Streamlit spinner to indicate processing:

if query:
    with st.spinner('Fetching recommendations for you...'):
        response = pipeline.recommend(query)
        st.markdown('### Recommendations')
        st.write(response)


Here, the pipeline's "recommend" method is called with the user query, and the recommendations are displayed on the app using "st.write" or "st.markdown" to format the heading.

In summary, the app works as follows:

We import Streamlit and the prediction pipeline.

Environment variables are loaded.

Page configuration is set.

A cached function initializes the anime recommendation pipeline only once.

A title and input box are displayed for the user.

When the user enters a query, a spinner is shown while the recommendations are fetched.

The recommendations are displayed in the UI.

To run the app, open a terminal and execute "streamlit run app/app.py". The first run may take slightly longer because the pipeline is being initialized and cached. After that, recommendations are fetched quickly.

You can try queries like "A light-hearted anime with school setting", "Similar anime to Naruto", or "Similar anime to Bleach". The app will fetch and display recommendations accordingly.

With this, your Streamlit application is complete! In the next video, we will create the Dockerfile, Kubernetes deployment files, and cover CI/CD, code versioning, and production deployment using Docker, Kubernetes, and cloud platforms.

**Summary:**

In this video, the focus is on creating the main application for the Anime Recommender System using Streamlit. A file named app.py is created inside the app directory. The application integrates the prediction pipeline, which loads the pre-built vector store as a retriever and uses the LLM along with the prompt template to generate anime recommendations based on user queries. Environment variables are loaded using load_dotenv() to ensure the API key and Hugging Face token are accessible, especially for deployment scenarios like Docker.

The Streamlit page configuration is set with a wide layout and a proper page title. A caching function init_pipeline() is defined using @st.cache_resource to initialize the pipeline only once, improving performance for repeated queries. The app displays a title and a text input box for the user to enter anime preferences. When the user submits a query, a Streamlit spinner indicates that recommendations are being fetched. The pipeline’s recommend method is then called, and the results are displayed in the UI using st.write and formatted headings.

In summary, the main application brings together the pipeline and the Streamlit interface, providing a user-friendly way to enter queries and receive anime recommendations in real-time. The app can be run using streamlit run app/app.py, with the first run taking slightly longer due to pipeline initialization, and subsequent queries fetching results quickly. This completes the user-facing application, ready for deployment with Docker, Kubernetes, and cloud platforms in the next steps.

**10. Dockerfile , Kubernetes Deployment File and Code Versioning**

In this video, we will be creating the Dockerfile, the Kubernetes deployment file, and implementing code versioning using GitHub for our anime recommender project.

First, let’s start with the Dockerfile. In your root directory, create a file named "Dockerfile". This file will allow us to containerize our project so it can be deployed on Kubernetes or any cloud platform. You don’t need to write it from scratch; you can copy a pre-written Dockerfile from the GitHub repository linked in the resources section.

Here’s the breakdown of the Dockerfile:

We use a base image "python:3.11-slim".

Essential environment variables are defined for production purposes.

We set up a working directory inside the container using "WORKDIR /app".

Dependencies are installed by copying the project files into the container and running "pip install -e . --no-cache-dir". The --no-cache-dir ensures a fresh installation without using cached packages, avoiding conflicts with old dependencies.

The Dockerfile exposes port 8501, which is the default Streamlit port.

The command to run the app is defined as "streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true".

Explanation of the command arguments:

--server.port 8501 specifies the port.

--server.address 0.0.0.0 allows the app to be accessible externally.

--server.headless true prevents Streamlit from opening a browser automatically, which is important for production environments on Linux machines.

Next, let’s create the Kubernetes deployment file. In your project directory, create a YAML file, e.g., "lm_ops_k8s.yaml". A Kubernetes deployment file typically consists of two main sections: deployment and service. You can also create them separately, but in this project, we’ll combine them into a single YAML file.

Here’s a summary of the important configurations in the Kubernetes YAML file:

Deployment:

replicas: 1 (number of pod instances; can be increased depending on your server capacity).

containers: define the container name and image name.

imagePullPolicy: IfNotPresent ensures Kubernetes uses the local Docker image if available.

Environment variables are injected as Kubernetes secrets, so sensitive information like API keys are not pushed to GitHub.

Service:

type: LoadBalancer exposes the application to the internet. Other options are ClusterIP (internal) or NodePort (external).

Ports configuration: port: 80 (external) and targetPort: 8501 (Streamlit container port).

The service links to the deployment using the same app name.

This setup allows the application to run in a scalable, production-ready Kubernetes environment while keeping secrets secure.

Finally, let’s handle code versioning using GitHub.

In the root directory, create a ".gitignore" file and list files/folders you don’t want to push, such as:

env/ (environment variables)

__pycache__/ (cached Python files)

logs/ (log files)

Initialize a Git repository:

git init
git branch -M main
git remote add origin <your_github_repo_url>
git add .
git commit -m "Initial commit"
git push origin main


This pushes your project to GitHub, excluding files specified in ".gitignore", so your API keys and local dependencies remain secure.

Additionally, we have an updated CSV file, "anime_updated.csv", which contains the anime dataset with columns like title, overview, and genre. Make sure this file is included in your repository.

With this, you have successfully created the Dockerfile, Kubernetes deployment file, and implemented code versioning on GitHub. Your project is now ready for production deployment.

**Summary:**

In this video, the focus is on preparing the Anime Recommender Project for production deployment using Docker, Kubernetes, and GitHub for code versioning. First, a Dockerfile is created in the root directory to containerize the project. The Dockerfile uses the base image python:3.11-slim, sets up a working directory /app, installs dependencies using pip install -e . --no-cache-dir, exposes port 8501, and defines the command to run the Streamlit app with --server.port 8501, --server.address 0.0.0.0, and --server.headless true. These configurations ensure that the app runs in a headless, production-ready environment and is externally accessible.

Next, a Kubernetes deployment YAML file, e.g., lm_ops_k8s.yaml, is created. This file defines both the deployment and service. The deployment specifies the number of replicas, container details, and image pull policies, while sensitive environment variables like API keys are injected securely as Kubernetes secrets. The service exposes the application using a LoadBalancer on port 80, linking it to the container’s Streamlit port 8501. This setup allows scalable deployment of the app in a production-ready Kubernetes cluster.

Finally, code versioning is implemented using GitHub. A .gitignore file is created to exclude sensitive files such as environment variables, Python caches, and logs. The repository is initialized with git init, a main branch is set, and the project is pushed to GitHub using standard Git commands. The updated CSV dataset anime_updated.csv is included in the repository. With these steps, the project is now fully containerized, deployable on Kubernetes, and version-controlled, making it ready for production use.

**11. GCP VM Instance Setup with Docker Engine , Minikube and Kubectl**

In this video, we will set up a VM instance for our project. We will be using Google Cloud Platform (GCP), although you can also use AWS or Azure. GCP is recommended because it is cost-efficient and easy to implement.

First, log in to your Google Cloud account, search for VM instances, and create a new instance. You can keep the default name, region, and zone. For the machine type, select E2 Standard and allocate 16 GB memory with 4 CPUs. This configuration ensures that Minikube, which requires at least 2 CPUs, can run efficiently while leaving resources available for other computations. You will also receive $300 free credits on GCP, which makes it affordable for testing.

Next, configure the OS and storage:

Select Ubuntu 24.04 LTS (x86_64). LTS stands for long-term support, which is stable and reliable. Avoid the newer 25 version as it may be unstable.

Allocate 150 GB of disk space, which is sufficient for our needs.

In the networking section, enable all three checkboxes and enable IP forwarding. Leave other options as default and create the VM instance. Once it is running (green tick appears), connect using SSH, which opens a browser-based terminal. Authorize the connection and clear the screen with the clear command.

Step 1: Install Docker Engine

We need Docker first because Minikube depends on it.

Go to the official Docker installation page for Ubuntu and copy the commands for setting up Docker’s repository. Paste and run them in your VM terminal.

Install Docker using the commands and authorize with yes.

Test Docker installation with:

sudo docker run hello-world


This works, but we don’t want to run Docker with sudo every time.

Follow the post-installation steps from Docker documentation:

Add your user to the Docker group

Reload the group membership

Test Docker again:

docker run hello-world


Now Docker runs without sudo.

Enable Docker to start automatically on boot using systemd commands, ensuring Docker service is active whenever the VM starts.

You can verify the installation with:

docker version

Step 2: Install Minikube

Minikube will create a local Kubernetes cluster on your VM and relies on Docker to run containers.

Go to the Minikube Linux installation page and copy the stable binary commands. Paste them one by one in your VM terminal.

Start the Minikube cluster:

minikube start


This will initialize your local Kubernetes cluster.

Step 3: Install kubectl

kubectl is the command-line tool to manage Kubernetes clusters.

Go to the official Kubernetes documentation, select Linux, and copy the installation commands. Paste them in the terminal.

On Ubuntu, an easier approach is to install kubectl via Snap:

sudo snap install kubectl --classic


Verify installation:

kubectl version --client


At this point, your VM instance is ready with all necessary tools:

Docker Engine – for building and running container images

Minikube – for running a local Kubernetes cluster

kubectl – for managing Kubernetes deployments

Everything is installed, and your Minikube cluster is running. Docker is set up and can run containers without sudo. This VM is now ready for deploying your Dockerized anime recommender application.

This completes the VM setup video. Next, we will move on to deploying the application on this VM using Docker and Kubernetes.

**Summary:**

In this video, we set up a virtual machine (VM) instance on Google Cloud Platform (GCP) to prepare for deploying the Anime Recommender application. Although AWS or Azure could also be used, GCP is recommended for its cost-efficiency and ease of implementation. After logging into GCP, we create a new VM instance with the default name, region, and zone. We select an E2 Standard machine type with 16 GB memory and 4 CPUs, ensuring sufficient resources for running Minikube alongside other processes. For the operating system, we choose Ubuntu 24.04 LTS (x86_64) for stability and allocate 150 GB of disk space. In networking, all checkboxes are enabled, including IP forwarding. Once the VM is running, we connect via SSH to access a browser-based terminal.

The first step is to install Docker Engine, which is required by Minikube. Following the official Docker installation instructions for Ubuntu, we set up the Docker repository, install Docker, and authorize post-installation steps to allow running Docker without sudo. Docker is also enabled to start automatically on boot and verified using docker version. Next, we install Minikube, which creates a local Kubernetes cluster on the VM and relies on Docker to run containers. Using the stable Minikube binaries, we start the Minikube cluster. Finally, we install kubectl, the Kubernetes command-line tool, either via official Linux installation commands or Snap, and verify the installation.

At this point, the VM is fully set up with all necessary tools: Docker Engine for container management, Minikube for running a local Kubernetes cluster, and kubectl for managing Kubernetes deployments. Docker is configured to run without sudo, and the Minikube cluster is operational. This VM is now ready to deploy the Dockerized Anime Recommender application in a Kubernetes environment, completing the VM setup for production deployment.

**12. GitHub Integration with Local and VM**

In this video, we will integrate GitHub with our VM instance.

As you saw in the previous video, we have already done the initial setup:

Our code has been pushed to GitHub.

We have created the Dockerfile and the Kubernetes deployment file.

Our VM instance is created, and we have installed Docker, Minikube, and kubectl on it.

Now, the next step is to connect your GitHub repository with both your local machine (VSCode) and your VM instance, so that changes made locally can be pushed to GitHub and then pulled to the VM automatically.

Step 1: Clone the GitHub Repository to the VM

Copy your repository URL from GitHub.

On your VM terminal, run:

git clone <your-github-repo-url>


Navigate into the cloned repository:

cd <repository-folder-name>


Run ls to verify all files are present (Dockerfile, deployment files, app code, setup scripts, etc.).

Step 2: Connect Local Repository to GitHub

To ensure changes made on your local machine can sync with GitHub:

Configure Git with your email and username:

git config --global user.email "your-email@example.com"
git config --global user.name "your-github-username"


Add, commit, and push changes from your local machine:

git add .
git commit -m "Commit message"
git push origin main


Important: GitHub no longer accepts your normal password for HTTPS authentication. You need to generate a Personal Access Token (PAT):

Go to GitHub → Profile → Settings → Developer Settings → Personal Access Tokens → Tokens (Classic) → Generate New Token.

Provide a name, select permissions (repo, workflow, admin:repo_hook), and generate the token.

Use this token as your password when pushing from Git.

Step 3: Pull Changes on VM Instance

Whenever you push new changes from your local machine, pull them on the VM using:

git pull origin main


Example workflow:

Create a test file on your local machine: test.py

Add, commit, and push:

git add test.py
git commit -m "Add test file"
git push origin main


On the VM, pull the changes:

git pull origin main
ls


You will now see test.py in the VM repository.

If you delete the file locally, commit and push again, then pull on the VM, the deletion will reflect automatically.

This process interlinks your local machine, GitHub repository, and VM instance, allowing seamless synchronization. This workflow can be applied to all future projects.

Now your VM is ready to always stay in sync with your GitHub repository, ensuring that the latest code is deployed and ready for Docker/Kubernetes tasks.

**Summary:**

In this video, we integrated GitHub with our VM instance to enable seamless synchronization between the local machine, GitHub repository, and the VM. Starting from the previous setup, we already had the code pushed to GitHub, the Dockerfile and Kubernetes deployment file ready, and the VM configured with Docker, Minikube, and kubectl. The first step is to clone the GitHub repository onto the VM by copying the repository URL and running git clone <repository-url> in the VM terminal. After navigating into the repository folder, we can verify that all files, including Dockerfile, deployment scripts, and application code, are present.

Next, we ensure the local machine is connected to GitHub. This involves configuring Git with the user’s email and username, and then using git add, commit, and push commands to synchronize local changes. Since GitHub no longer supports password authentication, a Personal Access Token (PAT) is required for HTTPS pushes. The PAT is generated in GitHub under Developer Settings, with appropriate permissions such as repo, workflow, and admin hooks, and is used as a password when pushing changes.

Finally, to keep the VM repository in sync, any updates pushed from the local machine can be pulled on the VM using git pull origin main. For example, creating a new file locally, committing, and pushing it to GitHub allows it to appear on the VM after pulling changes. Similarly, deletions or modifications in the local repository are reflected on the VM after a pull. This workflow ensures that the local development environment, GitHub, and VM instance remain synchronized, allowing the latest code to be deployed efficiently for Docker and Kubernetes tasks, and can be applied to future projects for smooth version control and deployment.

**13. GCP Firewall Rule Setup**

In this video, we will be setting up a firewall rule for our VM instance.

Step 1: Open Firewall Settings

Go to your Google Cloud Console and search for Firewall in the search bar.

Open the Firewall page in a new tab.

Step 2: Create a Firewall Rule

Click Create Firewall Rule.

Give your firewall rule a name (any name is fine).

Keep the Network set to default.

Set Direction of traffic to Ingress.

Set Action on match to Allow.

For Targets, select All instances in the network.

Set the Source filter to IPv4 ranges.

Enter the Source IP range as:

0.0.0.0/0


This will allow traffic from any IP address.

Step 3: Allow All Ports

Under Protocols and ports, select Allow all.

Click Create to finalize the firewall rule.

Once the rule is created, your VM instance will allow incoming traffic from all IPs on all ports. This is essential when you want your services, such as Docker containers or Kubernetes deployments, to be accessible externally.

**Summary:**

In this video, we set up a firewall rule for our VM instance on Google Cloud Platform to allow external access to our services. First, we navigated to the Firewall page in the Google Cloud Console and clicked on “Create Firewall Rule.” We gave the rule a name, kept the network as default, and set the direction of traffic to Ingress with the action set to Allow. For the targets, we selected all instances in the network, and for the source filter, we chose IPv4 ranges, entering 0.0.0.0/0 to allow traffic from any IP address. Under protocols and ports, we selected Allow all to ensure that all incoming ports are open. After clicking Create, the firewall rule was finalized, enabling the VM to accept incoming traffic on all ports from any IP. This setup is crucial for making services such as Docker containers and Kubernetes deployments externally accessible.

**14. Deployment of App on the Kubernetes**

In this video, we will be building our Docker image and deploying our app on the VM instance inside our Kubernetes cluster using Minikube.

Step 1: Point Docker to Minikube

Run the following command on your VM instance:

eval $(minikube -p minikube docker-env)


This ensures that Docker builds images specifically for the Minikube environment.

Step 2: Build the Docker Image

Make sure you are inside your GitHub repository where the Dockerfile exists.

Run the Docker build command:

docker build -t lm-ops:latest .


You can change the image name according to your preference.

This process may take a few minutes depending on your project size.

To verify, run:

docker images


You should see your lm-ops:latest image with its size listed.

Step 3: Inject Environment Variables into Kubernetes

Copy your API keys (e.g., Grok API, Hugging Face token) into a notepad.

Create a Kubernetes secret by running a command like:

kubectl create secret generic lm-secrets \
--from-literal=GROK_API_KEY=<your_grok_api> \
--from-literal=HUGGING_FACE_KEY=<your_hf_token>


In your deployment YAML, reference the secrets under envFrom::

envFrom:
  - secretRef:
      name: lm-secrets

Step 4: Apply the Kubernetes Deployment

Apply your deployment and service YAML file:

kubectl apply -f LM_ops.yaml


This will create both the deployment and the service (since both are defined in the YAML).

Step 5: Verify the Deployment

Check the status of your pods:

kubectl get pods


You should see lm-ops running with the number of replicas you defined (e.g., 1).

Step 6: Setup Minikube Tunnel for External Access

Run:

minikube tunnel


Keep this terminal dedicated to the tunnel. You will need a separate SSH session for other commands.

Open a new terminal to your VM if needed.

Step 7: Access Your App

Get the external IP of your service:

kubectl get service lm-service


Open your browser and navigate to:

http://<external-ip>:8501


Your Streamlit app should now be running externally.

Step 8: Test API Integration

Try a sample query like “anime similar to Naruto”.

Verify that both Grok and Hugging Face APIs are working correctly.

Once you see the app running and APIs working, your Docker + Kubernetes deployment on Minikube is successful.

**Summary:**

In this video, we deployed our Anime Recommender app on a VM instance inside a Kubernetes cluster using Minikube. First, we configured Docker to point to Minikube by running eval $(minikube -p minikube docker-env), ensuring that Docker images are built specifically for the Minikube environment. Next, we built the Docker image using docker build -t lm-ops:latest . and verified it with docker images. To securely provide API keys for Grok and Hugging Face, we created a Kubernetes secret using kubectl create secret generic lm-secrets and referenced it in our deployment YAML under envFrom. We then applied the deployment and service configuration with kubectl apply -f LM_ops.yaml and verified the pods with kubectl get pods. To allow external access, we ran minikube tunnel and obtained the external IP of our service using kubectl get service lm-service. Finally, we accessed the app in a browser at http://<external-ip>:8501 and tested the API integration with sample queries like “anime similar to Naruto”. The successful response confirmed that the Dockerized app, integrated with Grok and Hugging Face APIs, is running smoothly in the Kubernetes cluster.

**15. Monitoring Kubernetes using Grafana Cloud**

In this video, we will be setting up Grafana Cloud monitoring for our project. This is the main part of the project where we monitor our Kubernetes cluster and applications.

Step 1: Create a Namespace for Monitoring

Connect to a new terminal on your VM instance (since other terminals are in use for Minikube tunnel and app port forwarding).

Check existing namespaces:

kubectl get ns


Create a new namespace called monitoring:

kubectl create namespace monitoring


Verify:

kubectl get ns


You should see both default and monitoring namespaces.

Step 2: Install Helm

Grafana Cloud deployment requires Helm.

Search for “Install Helm” online and go to the official Helm site.

Follow the commands under the “From Script” section:

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash


This will install Helm on your VM instance.

Step 3: Create a Grafana Cloud Account

Go to Grafana Cloud
 and create an account.

You get a 14-day free trial which is sufficient for this setup.

Launch Grafana Cloud after logging in.

Step 4: Configure Kubernetes Observability

Go to Observability → Kubernetes → Start Sending Data.

Install the backend agent as instructed.

Provide your cluster name and namespace:

Cluster Name: minikube (default for Minikube clusters)

Namespace: monitoring (the namespace we created)

Create a new access token for Helm deployment (e.g., Grafana token).

Copy and save the token safely.

Step 5: Deploy Monitoring Resources via Helm

Grafana provides a Helm chart for deployment.

On your VM, create a values.yaml file:

vi values.yaml


Copy the Helm chart content into values.yaml after removing unnecessary parts like EOF and extra headers.

Save the file:

:wq!


Deploy the Helm chart:

helm upgrade --install grafana-monitoring -n monitoring -f values.yaml grafana/grafana


Verify deployment:

kubectl get pods -n monitoring


All Grafana pods should be running.

Step 6: Verify Grafana Cloud Monitoring

Go to your Grafana Cloud dashboard.

Refresh the page to see your Kubernetes cluster metrics.

You should see:

Namespaces: 5 (default + monitoring + others)

Workloads: 13

Pods: 17

Containers: 21

Your deployed LM ops application and other Minikube resources will be visible.

Step 7: Explore Monitoring Features

View nodes, workloads, pods, and containers in Grafana.

Create alerts for specific metrics if needed.

Monitor each namespace and pod separately.

Use debug metrics like kube_pod_info to get detailed container-level data.

With this, your Kubernetes cluster is now connected to Grafana Cloud, and you can monitor all workloads and resources graphically.

Make sure to clean up resources after your monitoring to avoid unnecessary costs.

**Summary:**

In this video, we set up Grafana Cloud monitoring for our Kubernetes cluster and deployed applications. First, we created a new namespace called monitoring on the VM instance using kubectl create namespace monitoring to isolate monitoring resources. Next, we installed Helm via the official script to manage Kubernetes deployments. After that, we created a Grafana Cloud account, which provides a 14-day free trial, and configured Kubernetes observability by providing the cluster name (minikube) and the monitoring namespace. We generated an access token for Helm deployment and created a values.yaml file with Grafana Helm chart configuration. Using helm upgrade --install grafana-monitoring -n monitoring -f values.yaml grafana/grafana, we deployed Grafana monitoring resources to the cluster. Deployment was verified with kubectl get pods -n monitoring, ensuring all Grafana pods were running. Finally, in the Grafana Cloud dashboard, we could visualize our cluster metrics, including namespaces, workloads, pods, and containers. The setup allowed monitoring of all Minikube resources, workloads, and the LM ops application, with features to create alerts, debug metrics, and observe container-level data. This setup ensures full observability of the Kubernetes cluster while maintaining a separate namespace for monitoring.

**16. Cleanup Process**

Now let’s talk about the cleanup process for this project.

Step 1: Delete Your VM Instance

Go to your VM instance in Google Cloud.

Select the VM you created for this project and click Delete.

As soon as you delete your VM instance, your Kubernetes cluster will also get disabled automatically. This means that all monitoring on Grafana Cloud will stop.

Step 2: Verify Grafana Cloud

After the VM is deleted, open your Grafana Cloud dashboard.

You’ll notice that your Kubernetes monitoring is no longer active, and the interface will show the initial “Start sending data” steps again.

This confirms that disabling your Kubernetes cluster automatically disables its Grafana monitoring.

Step 3: Final Notes

This concludes our project. You now have hands-on experience with:

Setting up a VM instance

Installing Docker, Minikube, and kubectl

Deploying an application on Kubernetes

Integrating Grafana Cloud for monitoring

Grafana Cloud is paid after the 14-day free trial, but the cost is generally manageable. In most companies, the company handles the subscription, so you don’t need to worry about payments.

During your trial, you can explore:

Metrics monitoring

Alerts

Dashboards

Advanced Kubernetes monitoring

With this our project is completed

**Summary:**

In this video, we covered the cleanup process for the Anime Recommender project. First, we deleted the VM instance in Google Cloud that was used for hosting the Kubernetes cluster. Deleting the VM automatically disables the Kubernetes cluster, which in turn stops all Grafana Cloud monitoring. To confirm, we checked the Grafana Cloud dashboard, where the monitoring data was no longer active, and the interface reverted to the initial “Start sending data” steps. This process ensures that all cloud resources and running workloads are properly cleaned up, preventing unnecessary costs. Finally, this concludes the project, where we gained hands-on experience in setting up a VM, installing Docker, Minikube, and kubectl, deploying an application on Kubernetes, and integrating Grafana Cloud for monitoring. While Grafana Cloud’s free trial lasts 14 days, it provides full access to metrics, dashboards, alerts, and advanced Kubernetes monitoring features, allowing a complete end-to-end deployment experience.












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

**In Utils created a ImageHandler.py file**

We will be getting input from user as a image; We have to do some transformation, so we are using this

(i) We used OpenCV (cv2); From io import BytesIO - Deals with In Memory Operations (io means Input Output)

## Then we used numpy for arrays working; Images are made of arrays

(ii) Creating the main Function - def process_image(image_file):
    
(iii) **Creating for Temporary Storage; Let's say user creates an image, we can't store everything in Local; When user uploads it will store for sometime in form of cache files; It stores temporarily** - in_memory_file = BytesIO()

(iv) **Storing the image in temporary storage created** - image_file.save(in_memory_file)

(v) **We have to get byte data of that image, we can't directly deal with images; It retrieve entire contents of BytesIO Object in form of bytes; It will fetch all content that we stored in form of bytes** - image_bytes = in_memory_file.getvalue()

(vi) **We can't work with Bytes too; So we are converting it to Numpy array** - nparr = np.frombuffer(image_bytes,np.uint8)

#### OpenCV Works with NumPy array only; It can't work with Bytes Data too

### Still this NumPy array is not compatible with OpenCV Format; So we have to convert it still; 

(vii) **It is decoding array and converting it to Image that is compatible with OpenCV, so it can be detected with OpenCV (OpenCV friendly format); User will upload RGB Image and we have converted it to Grayscale image,as it is easier to work with** - img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

### (viii) **We are converting here; Why not RGB to Gray; Because when we get a Image from Internet it is BGR and not RGB** gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Also grayscale images are best for face detection, so we are converting it to Grayscale

### We will use a Pre-Trained Computer Vision Model; By default it is inside Computer Vision Library; Inside OpenCV it is stored; 

### Technique is "Haar Cascade"; That is a pre-trained model used to detect Human Face; We will detect only the front part and not rear part; If someone uploads rear part, it won't be able to detect

#### (ix) "Loading Pre-Trained Model" - face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(x) **To detect faces, using face_cascade; Giving gray to detect it on GrayScale; It will detect the faces and store it in faces variable
** - faces =face_cascade.detectMultiScale(gray,1.1,5)

(xi) **If no Faces are detected, i.e., No face in the Image, then it will return that particular bytes and will give it none; None to say that No Face is there** - if len(faces)==0:
        return image_bytes,None

(xii) **There might be cases of Multiple Faces too; We will detect one face and ignore rest, as it is easier to do; It detects the Maximum Face; Largest image is the main subject; Let's say out of 5 person, one person is standing in front, then that is the main subject and the model will detect that** - largest_face = max(faces,key=lambda r:r[2] *r[3])

(xiii) **Using Co-Ordinates to detect Position and Size of the Image; These are co-ordinates of Largest Face** - (x,y,w,h) = largest_face

### We are detecting the largest face and it is stored in largest_face; We are splitting the co-ordinates from Largest Face; We are getting particular x,y,w,h values so that we can deal with them separately; x,y - Positions; w,h - Width and Height of Image

(xiv) **We generally detect using Rectangular boxes, so we are drawing rectangle using these; We will draw rectangle using x,y,w,h; We will be creating that rectangle on our main image; It is not on our Grayscale image and it is on our original image; We gave (x1,y1), (x2,y2); To get in Rectangular boxes, RGB, it will be in Green Colour as G is highest; 3 - it is how much thicker we want - (0,255,0)** - cv2.rectangle(img, (x,y),(x+w , y+h) , (0,255,0),3 )

(xv) **Encoding the Image into JPG Format** - is_sucess , buffer = cv2.imencode(".jpg" , img)

### We decoded at (vii) and now we are encoding; Here encoding into .JPG format; First we converted image into OpenCV Format and here converting to JPG format

(xvi) **Finally Returning; We have stored in Buffer variable and now returning  it; We are returning Largest Face and also encoded image, we are returning in form of bytes, so that later we can convert it into Normal Image** - return buffer.tobytes(), largest_face

### Summary: Take an Image, store it in temporary memory, convert into NumPy array, and convert it into OpenCV compatible friendly image, then into grayscale image, loading a pre-trained model, use that to detect faces, take the largest face / main subject in that image, detect co-ordinates of face, draw bounding box for that particular image, convert the OpenCV format image to JPG image, returnt the JPG image in Bytes format so that we can later convert it to Normal Image



4. **Created Celebrity Detector Code using Llama-4** - Creating celebrity_detector.py in utils; **Used GROQ and Meta Llama-4**

### We have a image perfectly right now and now we have to detect who is that celebrity 

**In utils folder created a file called "celebrity_detector.py**

(i) **Import os for environment variables; Request - Used to send request to our API; base64 - It is used to encode the image for the API Request, we can't send an Image to the API, we have to encode it first and then send it to API; The we will get answer fromt that API and then have to decode that image** - import os, import base64, import requests

(ii) **Created a celebrity class and defined __init__, Got the API Key from Environment variables; To give the API URL from which we have to get the answer (it is from GROQ --> API Reference --> get the POST API; Then we got the model from Groq; Still here we defined the environment variables** - class CelebrityDetector:, def __init__(self):, self.api_key = os.getenv("GROQ_API_KEY"), self.api_url = "https://api.groq.com/openai/v1/chat/completions"; self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"

(iii) **Creating an Identify Function; In previous notebook we encoded into Bytes format, so here too we are getting as Bytes format; This bytes format we need to convert it to base64 format; Because when we are working with anything with API, it should be in base64 format; So we are converting image bytes into base64 encoded string; We encoded and then decoded too, to get the perfect image** - def identify(self , image_bytes): encoded_image = base64.b64encode(image_bytes).decode()

(iv) **Defining Headers for API Key; Authorization bearer has our API Key; We are preparing headers with Authentication and content type; Content is that we are passing image in form of JSON; Bearer is that we are saying that hey we are sending Images** -         headers = {"Authorization" : f"Bearer {self.api_key}","Content-Type" : "application/json"}

**(v) Prompt (Copy pasted from ChatGPT); First thing is model (self.model), we are creating input prompt and model for the image; we have defined the model; Message we gave that it is coming from User; We also said content that who it is(Celebrity Recognition); We also said, if unknown, say unknown; First we said message and then attaching our image also because that Image LLM will understand; Giving Image URL, converting it to JPG format, attaching it in form of base64 encoded image (We already have the encoded image), we are passing it inside our prompt; Then we have max tokens - It is the limit or size of response when we give the query, it is in how many It is in how many tokens it will generate the output; Next is temperature; Higher the temperature, more creative our model is; More creative more hallucinative, if suppose we have given Virat Kohli, it can detect as other clebrity, so give it in between 0.2-0.5; Next we have to send POST Request to the API with the Prompt and the headers (We have header, image, prompt, API Key; Main thing is to send all this to API); So we sent it to "response" code; Response might be successful or it might have some error; responsecode=200, means we are saying it is API Request is successful and we got some output; We are storing it in result variable; Storing all the text output and storing in result variable; Finally, we are extracting name using extract_name function; Extract Name - Suppose we got the result, result will be in Content, we want to get the persons full name, so we wrote this function, it is iterating over each line and when it sees something as "full name", we will split it, it is iterating over data and fetching full name of that celebrity** - prompt = {"model": self.model,"messages": [
                {
                    "role": "user", 
                    "content": [ {
                            "type": "text",
                            "text": """You are a celebrity recognition expert AI. 
Identify the person in the image. If known, respond in this format:
- **Full Name**:
- **Profession**:
- **Nationality**:
- **Famous For**:
- **Top Achievements**:
If unknown, return "Unknown".
"""},



{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,    
            "max_tokens": 1024     
        }

response = requests.post(self.api_url , headers=headers , json=prompt)

if response.status_code==200:
            result = response.json()['choices'][0]['message']['content']
            name = self.extract_name(result)
            return result , name  
        return "Unknown" , ""  

def extract_name(self,content):
        for line in content.splitlines():
            if line.lower().startswith("- **full name**:"):
                return line.split(":")[1].strip()
        return "Unknown"  

### Summary: Importing Libraries, API Keys, environment variables, base64 for encoding with API, request to get request from API's, the created class celebrity_detector initialized api, uri, model (Vision Transformer), Identify method - Passing Image bytes in which in previous notebook we created Image bytes in the end, we can't deal with bytes because we are working with API's so we convert it to base64 and storing it into encoded_image, Next we prepared headers with API key authorization and content which will be returned in JSON, then we gave model (self.model), then we gave role as user,type and text saying what that model has to do; Then type is image_url, and image in JPG format, the encoded image in base64; Then we set our tokens and temperature; Then we got the response, where we got API authorization and content, headers and json, API will give results and that we stored in response; Then we checked if response is successful or not (If successful  200, if failed 404); We are extracting only text, because API will be returing as a Text only and we are extracting only content only; And then we are extracting the name; Finally, it is working on image, sending it to API, then getting the result, then getting content from the result of that particular person



5. **Created qaengine.py in utils - Question Answer Engine Code**

6. **Created Flask Backend Routes code using routes.py inside utils**

7. Writing the main code in "__init__.py" in Utils folder

### Here it will run in Flask in Local Host and Ports

8. Created a Docker File (Some of the OpenCV Library needs some dependencies)

9. Created Kubernetes Deployment yaml file

**We will inject API into Kubernetes Cluster**

10. Added .gitignore file for GitHub code versioning

### Pre-Deployment Checklist: Need to have Dockerfile, Kubernetes Deployment File, Code Versioning using GitHub

11. Deployment in GCP:

When using GCP for first time, enable Kubernetes Engine API,
Container Registry API,
Compute Engine API,
Cloud Build API,
Cloud Storage API,
IAM API

12. Creating a Kubernetes Cluster

13. Create a Artifact Registry

#### 14. Creating an Service account, because we want to give access to Kubernetes; We need to give our CI/CD Tool to give partial access to our Kubernetes

15. Create a Service Key and get as a JSON Format

16. Copy it to VS Code and change it's name to GCP Key; Add gcp file name in gitignore

### We need to fetch image from artifact registry and need to work with Kubernetes engine also; So we need GCP Key

#### We created Kubernetes Cluster, created Artifact Repository, created service account with permissions, created access key, downloaded as a JSON and stored in VS Code

17. Circle CI Codes: created "config.yml" file

### **Codes inside config.yml (Inside circleci Folder):**

18. Pulling the image from Docker, and installing it in our directory; Gave the repository to work with too

**Why use Google Image, instead of Python Image, because if we use Python Image or anything we need to install GCloud CLI Separately; So to reduce our work, we used Google Image; It's dependcies will be automatically installed**

19. Jobs - Checkout Code from GitHub, Build Docker Image, Deploy in Kubernetes Engine

### Whatever we have uploaded in GitHub, we are pulling into Docker Container; and using into Google Container; It is pulling everything, storing inside app (We use a Docker Executor here)

Second Job - Building Docker Image, Pushing it to Google Artifact Registry; We use same docker executor because we need docker to build and pushing the image

Setup remote docker - This command will let us build Docker Inside the circle CI; We have to write the code to use Docker Image inside Circle CI

Next two commands - Authenticating into GCP, Building and Pushing

## Then we had some encoded key; We have encoded GCP Key that we downloaded as a JSON into a special format known as Base64 and then stored inside environment variables inside CircleCI

We say that environment variable from Circle CI

We have again decoded and stored as JSON

## Why did we took a Long approach and not pushed directly?

Because we can't push GCP key to github or any source code management 

### We converted it to a Circle CI Environment Variable, after that we used it to recreate our field (We took that file from Local Repository, converted it to base64 Image, after that it gets converted to a simple one line code

## So we pasted in one Circle CI Environment Variables; We didn't expose it to any Github or any public repository; It is inside Circle CI Private Repository

## From that environment variable we decoded it again to GCP (Normal File); That's how we protected it from being exposed to a Public Platform

20. We have the GCP.Json and we can use it to activate our Service account; We use this GCP Key to activate Service Account (We have configure that region part alone to our need)

[Whole some part was for authentication purpose]

### We took our Key from Environment Variable, converted back into our file, decoded it back to GCP.JSON format and we used that file to activate the account

#### Then we autheticated our Docker using Particular Region

Now comes the Main Part of the Job

21. Building and Pushing Docker Image

**Docker Build Code is - It is the Google Repository Path Copied (Registry-Project_ID [We kept Project_ID inside our environment varibale] - Google Artifact Registry Name - Image Name (Whatever we give in the last is the Docker Image Name) [Whenever we change it make sure to change it inside Kubernetes Deployment file name too (To the kubernetes_deployment.yaml file)]**

We build Docker Image till here; Next we will push it;

Push Code format also Similar 

22. **Third Job - Image is in Google Artifact Registry; We use docker executor, doing the checkout; Setting up a Remote Docker, because we want to run Docker Commands, so we want to give permissions and setup Remote Docker**

### Again we have to authenticate with our GCP because, We need to deploy the Pushed Image; So we authenticated again to Google Cloud

23. **Next Configuration is for Google Kubernetes**

We wrote Gcloud Container cluster. passed Cluster name (Gave in Environment Variables and we will be fetching it), giving the region of Kubernetes Cluster; Then we give the Project ID (Same way we give in Environment Variables of Circle CI)

**Then we have authenticated our GCP and Kubernetes**

### Main Part is Deployment and we will Deploy using "kubernetes_deployment.yaml" file 

24. Gave the file name in code;

#### kubectl-rollout - Everytime new docker image is used each time pipeline runs 

We have to keep same app name "llmops-app" in both "kubernetes_deployment.yaml" file and "config.yml" file

**Image name also should match in both**

25. Next comes the Workflow code, where we define the workflow

###  We have defined the Jobs, but we didn't define in which pattern the Jobs should run (Whether to checkout, build, push; We didn't define that pattern in which each step has to be performed)

#### Pattern is, it has to run the checkout code, then build docker image (It can run only if the checkout code is fullfiled) and run, then deploy to Kubernetes (It can run only if Docker Image is successful)

#### This is what whole workflow of Circle CI CI/CD Pipeline


### Full CI/CD Deployment of Application on GKE:

First of all, we begin by converting the GCP key into base64 format. To do this, open VSCode and launch a Git Bash terminal. If Git CLI is installed, Git Bash comes pre-installed with it. You can simply search for Git Bash and open it. Once inside, type the command you have copied for base64 conversion. Before executing, make sure your GCP key file is present in the root directory. After pasting the command, press Enter. This will convert your GCP key into base64 format, essentially encoding it. The result is your encoded GCP key, which you can copy for further use.

Before proceeding, create an account on the CircleCI website. Visit the site, sign up, complete verification, and log in. After logging in, you will land on the CircleCI dashboard where you need to create a new project. Navigate to the “Projects” section and select an option to create a project. You can name it anything; for example, let’s call it “LM ops.” Next, click on “Set up a pipeline.” For the pipeline name, you can use “Build and Test,” or keep it as the default. After this, you must connect your GitHub repository. In some cases, it might already be connected, but if not, you will be prompted to authorize CircleCI to access GitHub. Once connected, you can see all your repositories.

Now, select your repository. Suppose your repo is named celebrity detector question and answer. Search for that name, select it, and then click on “Set up your config.” If you already have a .circleci/config.yml file in your GitHub repository, CircleCI will automatically detect it. However, if you don’t see this file yet, you need to push it first. Simply go to the command line, run git add, git commit, and git push. Once pushed, go back to CircleCI, reselect the repo, and CircleCI will detect the config.yml file automatically.

Next, proceed to set up your triggers. By default, CircleCI provides one trigger: every time you push something to your GitHub repo, the pipeline will run. This ensures your pipeline restarts automatically upon each push. While you can choose other triggers, in this case we will stick to “All pushes.” Once this is selected, review the setup: we have named the pipeline “Build and Test,” are using the existing configuration file, and selected the trigger for all pushes. Finish the setup to complete this step.

Now move to defining environment variables. Go to the “Settings” of your project, and on the left-hand menu you’ll find “Environment Variables.” Click on it, and start adding variables one by one. The first environment variable is your Google Cloud service key. Copy its name exactly as it appears in your config.yml file, paste it in the “Name” field (without the dollar sign), and set its value as the base64-encoded GCP key you generated earlier. Copy the entire encoded key from your terminal output carefully, ensuring there are no extra spaces at the beginning or end, and paste it as the value. Add this variable.

Next, add another environment variable for the project ID. Copy the name as given, and paste it into the name field. To get the project ID, go to your Google Cloud Platform (GCP) console, select your project, and copy the ID. Paste it as the value in CircleCI and save. Similarly, you can also create an environment variable for your repo name if needed, although in this case it’s optional.

After that, define the GKE cluster name as another environment variable. Go to your Kubernetes clusters in GCP, copy the cluster name, and use it as the value. Then, add the compute region as the next environment variable. Again, you can find the compute region when viewing your cluster details. Copy it, paste it in CircleCI, and save. At this point, you should have defined all required variables: Google Cloud service key, Project ID, GKE cluster, and Google compute region.

Even though the pipeline is set to run on all pushes, you may also want to trigger it manually. To do this, go to “Pipelines” in CircleCI, select your LM ops project, choose the branch (for example, main), and run the pipeline. This will manually start your pipeline. The pipeline will also continue to run automatically whenever you push new code. When triggered, the first step will be checking out your code, followed by building and deploying your Docker image to Google Kubernetes Engine.

One important note: while pasting the base64-encoded GCP key into CircleCI, ensure no random spaces are added at the start or end. Even a single space can cause an error. Be careful when copying. If all is done correctly, your pipeline will complete successfully and deploy your Docker image to your GKE cluster.

After deployment, navigate to GCP workloads. Initially, you might encounter an error saying the app does not have minimum availability. If you open the details, you may find the reason is a missing secret—such as lm_ops_secret. For example, inside your Kubernetes deployment file, you might have defined secrets like an API key. Since Kubernetes cannot find this secret, it throws a container configuration error. To fix this, go back to your cluster in GCP and connect to it via kubectl. Use Cloud Shell to run the connection command. GCP may automatically generate the gcloud container clusters get-credentials command with your cluster name, region, and project ID, but if not, you can manually provide these details.

Once connected, inject your missing secret using a command like kubectl create secret. Replace the placeholder with your actual API key. You can copy the API key from your CircleCI environment variables, paste it into the command, and execute it in Cloud Shell. If successful, Kubernetes will confirm the secret has been created. This ensures that your application will no longer throw the missing secret error.

Now return to CircleCI and trigger the pipeline again. Once it finishes successfully, revisit your Kubernetes workloads. This time, you should see your application deployed without errors. At first, it might briefly show minimum availability issues, but wait for a few minutes since trial versions of GCP often have limited resources, causing slight delays in provisioning. Once stabilized, you will see an endpoint listed under the load balancer section of your workload. Opening this endpoint will display your running application.

You can now test the application. For instance, upload an image to detect a celebrity. The app might identify someone like Salman Khan and provide details such as his profession and age. Similarly, you can test with other images—like Robert Downey Jr.—and ask questions about their roles. The application should respond correctly, demonstrating that it is functioning as expected.

Finally, remember to perform cleanup to avoid unnecessary charges. Go back to GCP, open your Kubernetes cluster, and delete it by confirming the cluster name. Also, delete your artifact repository where Docker images are stored. While you won’t be charged for leaving service accounts or CircleCI projects, you can delete them too if desired. The most critical cleanup steps are deleting the Kubernetes cluster and the artifact repository.

In summary, we deployed an application on Google Kubernetes Engine using CircleCI, tested it successfully, and then performed the necessary cleanup to manage costs. This completes the project and ensures you not only know how to deploy but also how to maintain resources responsibly.


### Project Notes from Udemy:

**1. Introduction to the Project**

So as you’ve already seen in the demo video, here’s what we are building:

We are creating an AI Celebrity Detector and Question Answer System.

User uploads an image of a celebrity.

Our model detects the celebrity’s name and provides some basic details.

The user can then ask questions about that celebrity using our language model (LM).

This is the overall idea of the application.

Main Components / Tech Stack

Here are the key technologies we are using:

Google Cloud (GCP) – For hosting and managing our services.

Kubernetes (GKE) – For deploying and scaling our application in a cluster.

CircleCI – Cloud-based CI/CD tool for automating deployment.

Flask – Backend framework to handle processing and API calls.

HTML & CSS – Frontend interface for user interaction.

OpenCV (Python) – For image processing (grayscale conversion, resizing, drawing bounding boxes).

Grok LM & LLaMA – For celebrity recognition and answering questions.

Docker & Google Artifact Registry (GAR) – For containerizing the app and storing Docker images.

Why These Technologies?

OpenCV: Convert RGB images to grayscale, detect faces, scale images, and highlight detected faces.

Grok LM + LLaMA: Handle image-based celebrity recognition and language-based question answering.

CircleCI: Fetches code from GitHub and automates CI/CD. Cloud-based, unlike Jenkins.

Flask + HTML/CSS: Flask handles backend logic, HTML/CSS creates an interactive frontend.

Docker + GAR: Containerize the app and store images securely for deployment.

GKE: Predefined Kubernetes cluster on Google Cloud to deploy your containerized app.

Project Workflow

Project & API Setup

Create a virtual environment.

Set up logging and exception handling.

Install required libraries (OpenCV, Flask, LangChain, etc.).

Define project structure and generate API keys for LLaMA.

Image Handler

Convert uploaded images to grayscale and numpy arrays.

Preprocess images for celebrity detection.

Celebrity Detector

Detect the celebrity from the processed image.

Retrieve basic details about the celebrity.

QA Engine

Answer user questions about the detected celebrity.

Flask Routes

Define all backend routes for image upload, celebrity detection, and question answering.

Main Application

Integrate Flask backend with HTML/CSS frontend.

Make the application interactive and functional.

Docker & Kubernetes

Dockerfile: Containerizes the application.

Kubernetes deployment file: Configures replicas, ports, and services for deployment on GKE.

GitHub & CI/CD

Push code to GitHub (source code management).

CircleCI Pipeline:

Checkout Stage: Fetch code from GitHub.

Build & Push Image: Build Docker image and push to GAR.

Deploy to GKE: Deploy Docker image to the Google Kubernetes Engine cluster.

Outcome

Once deployed, the application allows users to:

Upload a celebrity image.

Detect the celebrity and get details.

Ask any questions about the celebrity.

Fully automated CI/CD pipeline ensures smooth updates and deployments.

This concludes the workflow and tech overview for our AI Celebrity Detector & QA system.

Hope you enjoy the project and are ready to move forward to the implementation!

**Summary:**

As demonstrated in the project demo, we are building an AI Celebrity Detector and Question Answer System. The application workflow is straightforward: the user uploads an image of a celebrity, the model detects the celebrity’s name, provides basic details, and then the user can ask questions about the celebrity using a language model (LM).

Main Components / Tech Stack:

Google Cloud (GCP): Hosting and managing services.

Kubernetes (GKE): Deploying and scaling the containerized application.

CircleCI: Automating CI/CD processes for seamless deployment.

Flask: Backend framework to handle API calls and processing.

HTML & CSS: Frontend interface for user interaction.

OpenCV (Python): Image preprocessing (grayscale conversion, resizing, bounding boxes).

Grok LM & LLaMA: Celebrity recognition and language-based question answering.

Docker & Google Artifact Registry (GAR): Containerizing the app and storing images securely.

Why These Technologies:

OpenCV: Converts images to grayscale, detects faces, scales images, and highlights detected faces.

Grok LM + LLaMA: Handles celebrity recognition and answers questions about them.

CircleCI: Cloud-based CI/CD automation, fetching code from GitHub.

Flask + HTML/CSS: Flask handles backend logic, HTML/CSS provides an interactive frontend.

Docker + GAR: Containerizes the application and stores images securely.

GKE: Deploys and manages containerized applications in a scalable Kubernetes cluster.

Project Workflow:

Project & API Setup: Create virtual environment, set up logging and exceptions, install required libraries, define project structure, generate API keys.

Image Handler: Convert uploaded images to grayscale and numpy arrays, preprocess for celebrity detection.

Celebrity Detector: Detect the celebrity from the processed image and retrieve basic details.

QA Engine: Answer user questions about the detected celebrity.

Flask Routes: Define backend routes for image upload, celebrity detection, and question answering.

Main Application: Integrate Flask backend with HTML/CSS frontend for interactivity.

Docker & Kubernetes: Containerize the app with Dockerfile, define deployment and services for GKE.

GitHub & CI/CD: Push code to GitHub, CircleCI pipeline automates build, push to GAR, and deploy to GKE.

Outcome:
Once deployed, users can upload a celebrity image, detect the celebrity, view their details, and ask questions interactively. The fully automated CI/CD pipeline ensures smooth updates and deployment.

This provides a complete end-to-end workflow and tech overview for the AI Celebrity Detector & QA system, setting the stage for implementation.

**2. Project and API Setup ( Groq )**

This is the Project Setup video for our Celebrity Detector and Question Answer System.

Step 1: Create Project Folder

Create a new folder named:

Celebrity Detector and Question Answer System


Open it in VS Code:

Option 1: Right-click → Open in VS Code.

Option 2: Open terminal in that folder and run:

code .

Step 2: Set Up Virtual Environment

Open a new terminal in VS Code.

Create a virtual environment:

python -m venv env


Activate the virtual environment:

env\Scripts\activate


You’ll see (env) in your terminal, indicating successful activation.

Step 3: Create requirements.txt

Create a requirements.txt file in the root directory with the following libraries:

Flask – Backend framework

OpenCV-Python – Image processing

numpy – Handling image arrays

requests – HTTP requests

python-dotenv – Load environment variables

Save the file.

Step 4: Create setup.py

Create a setup.py file in the root directory.

Copy the code from the provided GitHub repository.

Set the project name, e.g., Celebrity Detector and QA System.

Purpose:

This file allows your app folder to be treated as a package.

Required for installing all dependencies and packages properly.

Step 5: Define Project Structure

Folders & Files:

Celebrity Detector and QA System/
│
├── app/               # Main application components
│   ├── __init__.py    # Marks this as a package
│   └── utils/         # Utility functions
│       └── __init__.py
│
├── templates/         # HTML files
├── static/            # CSS & JS files
├── requirements.txt
├── setup.py
└── .env               # Environment variables


app/ → Main backend logic

utils/ → Reusable utility functions

templates/ → Frontend HTML

static/ → CSS & JavaScript

__init__.py → Marks folder as a package

Step 6: Install Dependencies

Run the following command to install all requirements and set up packages:

pip install -e .


This triggers setup.py and installs all required libraries.

Step 7: Configure Environment Variables

Create a .env file in the root directory.

Add your Grok API Key for the LLaMA model:

GROK_API_KEY="your_api_key_here"


How to get the API Key:

Sign up on Grok Cloud (free account).

Navigate to the API Keys section.

Create a new key, name it celebrity, and copy it.

Paste it into your .env file.

Project Setup Summary

By the end of this video, you have:

Created project folder and opened it in VS Code.

Set up a virtual environment and activated it.

Created requirements.txt with necessary libraries.

Created setup.py to handle package installation.

Defined the project structure (app, utils, templates, static).

Installed all dependencies using pip install -e ..

Added environment variables for the API key in .env.

Now your project is ready for development, and you can move on to image handling and celebrity detection in the next video.

**Summary:**

In this video, we set up the foundation for the Celebrity Detector and Question Answer System project. First, a dedicated project folder named Celebrity Detector and Question Answer System was created and opened in VS Code to organize all project files systematically. A virtual environment was then set up and activated to isolate project dependencies and ensure a clean development setup.

Next, a requirements.txt file was created listing all essential libraries, including Flask for the backend, OpenCV for image processing, NumPy for handling image arrays, requests for HTTP requests, and python-dotenv for managing environment variables. Alongside this, a setup.py file was added to treat the project as a Python package, allowing for smooth installation of dependencies and proper project structure management.

The project structure was clearly defined with dedicated folders: app/ for the main backend logic, utils/ for reusable functions, templates/ for HTML frontend files, and static/ for CSS and JavaScript files. The __init__.py files marked the folders as packages, making it easier to organize and import modules. After defining the structure, all dependencies were installed using pip install -e ., which triggered setup.py and ensured all required libraries were correctly installed.

Finally, a .env file was configured to store sensitive information like the Grok API Key, which is necessary for interacting with the LLaMA model for celebrity recognition. This step ensures secure handling of API credentials. By the end of this setup, the project had a clean and organized structure, all necessary dependencies installed, and environment variables configured, making it ready for the next stages of development, including image handling, celebrity detection, and question answering.

**3.Image Handler Code with OpenCV**

In this video, we will create the Image Handler utility code. This will handle user-uploaded images for face detection.

Step 1: Create the File

Navigate to the utils directory.

Create a file named:

image_handler.py


Purpose:

Users upload images, but we cannot directly process them.

This utility will handle conversions, preprocessing, and temporary storage.

Step 2: Import Libraries
import cv2
from io import BytesIO
import numpy as np


Explanation:

cv2 → OpenCV library for image processing.

BytesIO → In-memory file handling (temporary storage).

numpy → Handle image arrays (images are arrays of pixel values).

Step 3: Define the Function
def process_image(image_file):


This function will process an uploaded image.

image_file → The image uploaded by the user.

Step 4: Create Temporary Memory
memory_file = BytesIO()
image_file.save(memory_file)
image_bytes = memory_file.getvalue()


Explanation:

Stores the uploaded image in memory temporarily.

Prevents saving thousands of images to local storage.

image_bytes contains the image in bytes format for processing.

Step 5: Convert to Numpy Array
np_array = np.frombuffer(image_bytes, np.uint8)
img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)


Convert byte data to a numpy array.

Convert the array to an OpenCV-compatible image.

Step 6: Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


Converts the original BGR image to grayscale.

Grayscale images are easier and faster for face detection.

Step 7: Load Pre-Trained Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)


Explanation:

Uses OpenCV’s Haar Cascade to detect frontal faces.

faces stores all detected faces in (x, y, w, h) format.

Step 8: Handle No Face or Multiple Faces
if len(faces) == 0:
    return image_bytes, None

largest_face = max(faces, key=lambda r: r[2]*r[3])
x, y, w, h = largest_face


No face: Return original image and None.

Multiple faces: Select the largest face (main subject).

Extract coordinates of the largest face (x, y, w, h).

Step 9: Draw Rectangle on Face
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)


Draws a green rectangle around the detected face on the original colored image.

Step 10: Encode Image as JPEG
success, buffer = cv2.imencode('.jpg', img)
return buffer.tobytes(), largest_face


Converts the OpenCV image to JPEG format.

Returns:

The image in bytes format (for later use/display).

The coordinates of the largest face (x, y, w, h).

Summary of Image Handler

Accept user-uploaded image.

Store it temporarily in memory.

Convert image from bytes → numpy array → OpenCV image.

Convert colored image → grayscale (for face detection).

Load pre-trained frontal face detector.

Detect faces in the image:

If no faces → return original image and None.

If multiple faces → detect only largest face.

Extract coordinates of the face (x, y, w, h).

Draw a bounding box on the face.

Encode image to JPEG bytes for output.

Return encoded image and face coordinates.

This utility function ensures the uploaded images are processed efficiently without storing them locally and prepares them for the next steps in your Celebrity Detector system.

**Summary:**

In this video, we created the Image Handler utility, which is responsible for processing user-uploaded images for face detection. The utility is implemented in a file named image_handler.py inside the utils/ directory. Its primary purpose is to handle image conversions, preprocessing, and temporary memory storage so that uploaded images can be processed efficiently without saving them permanently on disk.

We started by importing necessary libraries: OpenCV (cv2) for image processing, BytesIO for in-memory file handling, and NumPy (numpy) for working with image arrays. A function named process_image(image_file) was defined to handle the uploaded image. The image is first stored in memory using BytesIO, and its byte representation is extracted for further processing.

The byte data is then converted to a NumPy array and decoded into an OpenCV-compatible image. To simplify face detection and improve performance, the image is converted from color (BGR) to grayscale. A pre-trained Haar Cascade classifier is loaded to detect frontal faces. The function checks for the presence of faces: if none are detected, it returns the original image and None. If multiple faces are detected, the largest face (assumed to be the main subject) is selected. Coordinates (x, y, w, h) of the largest face are extracted.

For visualization, a green rectangle is drawn around the detected face on the original color image. Finally, the image is encoded as a JPEG and returned as bytes along with the coordinates of the largest face.

In summary, the Image Handler utility efficiently processes user-uploaded images by temporarily storing them in memory, converting them into OpenCV-compatible format, detecting the main face using a Haar Cascade classifier, drawing a bounding box, and returning the processed image and face coordinates. This prepares the images for the subsequent steps in the Celebrity Detector system.

**4. Celebrity Detector Code using Llama-4**

In this video, we’ll create our Celebrity Detector code.

In the previous video, you made the image handling code. Now you have a perfect image, and we’ll detect who that celebrity is.

We’ll implement this inside a file named detector.py under the utils folder.

Step 1: Import Libraries
import os       # For environment variables
import base64   # To encode images
import requests # To send API requests


OS → Load your API key from environment variables.

base64 → Encode your image before sending to the API.

requests → Send HTTP requests to the API endpoint.

Step 2: Create the Celebrity Detector Class
class CelebrityDetector:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        self.api_url = "YOUR_API_ENDPOINT_URL"
        self.model = "llama-for-maverick-17b-instruct"


api_key → Loaded from environment variables.

api_url → Endpoint URL from the API documentation.

model → The model to use for celebrity recognition.

Step 3: Define the identify Method
def identify(self, image_bytes):
    # Convert image bytes to base64
    encoded_image = base64.b64encode(image_bytes).decode()
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }
    # Prepare prompt
    prompt = {
        "model": self.model,
        "messages": [
            {
                "role": "user",
                "content": "You are a celebrity recognition expert..."
            }
        ],
        "image_url": encoded_image,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    # Send POST request
    response = requests.post(self.api_url, headers=headers, json=prompt)
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        name = self.extract_name(result)
        return result, name
    else:
        return "Unknown", "Unknown"

Key Points:

Image Conversion: APIs expect base64 images, not raw bytes.

Headers: Include your API key and content type (application/json).

Prompt: Instruct the model to identify the celebrity, returning Full Name, Profession, Nationality. If unknown, return "Unknown".

Temperature: Controls creativity. Lower = more accurate, less hallucination (0.2–0.5 recommended).

Max Tokens: Output limit for the API (1024 is sufficient).

Step 4: Define the extract_name Method
def extract_name(self, content):
    for line in content.splitlines():
        if line.lower().startswith("full name"):
            return line.split(":")[1].strip()
    return "Unknown"


This extracts the full name from the API response.

If it cannot find the name, it returns "Unknown".

Step 5: How it Works

Pass Image Bytes → Comes from the image handler in the previous video.

Convert to Base64 → Required for API requests.

Prepare Headers & Prompt → Set API key, model, content type, image, instructions.

Send POST Request → Hit the API endpoint.

Check Response → Status code 200 = success.

Extract Result → Fetch the text output.

Extract Name → Use extract_name() to get the celebrity’s full name.

Handle Errors → Return "Unknown" if something goes wrong.

In Short: The Celebrity Detector takes an image, sends it to the API, receives a textual result, extracts the celebrity's name, and returns it along with the full result.

**Summary:**

In this video, we created the Celebrity Detector utility, which identifies the celebrity in an uploaded image. This code is implemented in a file named detector.py under the utils/ folder. The main purpose of this utility is to take the preprocessed image from the Image Handler, send it to a celebrity recognition API, and return both the detailed API response and the celebrity’s full name.

We began by importing essential libraries: os to load environment variables (for the API key), base64 to encode images before sending them via API, and requests to handle HTTP POST requests to the celebrity recognition endpoint. A class named CelebrityDetector was created. In its constructor, we load the API key from environment variables, set the API endpoint URL, and define the model (in this case, "llama-for-maverick-17b-instruct") to use for celebrity recognition.

The core functionality is implemented in the identify method. It first converts the image bytes into a base64-encoded string, which is required by the API. Headers are prepared with the API key and content type (application/json). A prompt is created to instruct the model to identify the celebrity and return details such as Full Name, Profession, and Nationality. A POST request is sent to the API endpoint, and if the response is successful (status code 200), the textual output is extracted. The extract_name method parses this output to retrieve the celebrity’s full name. If no name is found or the API request fails, it returns "Unknown".

In summary, the Celebrity Detector works by taking an image from the user, converting it to a base64 format, sending it to a specialized API, and processing the response. It returns both the full textual description from the API and the extracted full name of the detected celebrity, providing a bridge between image recognition and further question-answering capabilities in the application.

**5. Question Answer Engine Code**

In this video, we’ll create a Question Answer Engine.

We’ve already detected the celebrity in the previous step. Now, this engine will allow us to ask questions about that celebrity.

We’ll implement this in utils/QA_engine.py.

Step 1: Import Libraries
import os       # To load environment variables
import requests # To send HTTP requests to API


OS → Load your API key.

requests → Send requests to the API endpoint.

Step 2: Create the QA Engine Class
class QAEngine:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        self.api_url = "YOUR_API_ENDPOINT_URL"
        self.model = "llama-for-maverick-17b-instruct"  # You can change this to another chat-capable model


api_key → Fetched from environment variables.

api_url → Same endpoint as in the celebrity detector.

model → LLM used for generating answers.

Step 3: Define ask_about_celebrity Method
def ask_about_celebrity(self, name, question):
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }
    # Prepare prompt
    prompt = f"""
    You are an AI assistant who knows a lot about celebrities.
    Answer questions about this celebrity concisely and accurately.
    Celebrity Name: {name}
    Question: {question}
    """
    # Prepare payload
    payload = {
        "model": self.model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,  # 0.3-0.5 recommended for accurate yet slightly creative responses
        "max_tokens": 512
    }
    # Send request to API
    response = requests.post(self.api_url, headers=headers, json=payload)
    # Check response
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Sorry, I couldn't find the answer."

Step 4: Key Points

Headers → Include API key for authorization and content type (application/json).

Prompt → Instructs the model to act as a celebrity expert and answer questions about the given celebrity.

Temperature → Controls creativity: higher temperature = more creative but may hallucinate.

Max Tokens → Output length limit (512 is usually sufficient for QA).

Response Handling → Extract the content from JSON response. Return an error message if the API fails.

Step 5: How it Works

Pass Celebrity Name and Question → Name comes from the Celebrity Detector.

Create Prompt → Concatenate the name and user question in a clear instruction.

Send API Request → Send payload to the LLM API endpoint.

Get Response → Extract the text output from the JSON response.

Handle Errors → Return a friendly message if something goes wrong.

In short:

The QA Engine allows you to ask any question about a detected celebrity. It takes the celebrity’s name and a user question, sends it to the LLM API, and returns a concise, accurate answer.

**Summary:**

In this video, we created the Question Answer (QA) Engine, which allows users to ask questions about the celebrity detected in the previous step. This functionality is implemented in utils/QA_engine.py. The main goal of this module is to take the celebrity’s name and a user-provided question, send them to a language model API, and return a concise, accurate answer.

We started by importing os to load environment variables (for the API key) and requests to handle HTTP requests to the API endpoint. A class named QAEngine was defined, where the constructor loads the API key from environment variables, sets the API endpoint URL, and specifies the model to use (in this case, "llama-for-maverick-17b-instruct").

The core functionality is in the ask_about_celebrity method. It takes the celebrity’s name and the user’s question, constructs a prompt instructing the model to act as a celebrity expert, and sends this prompt in a JSON payload to the API. The payload includes headers for authorization, the selected model, a temperature parameter to control creativity, and a max token limit to restrict output length. The API response is checked for success (status code 200), and the answer is extracted from the JSON content. If the request fails, a friendly error message is returned.

In summary, the QA Engine bridges the celebrity detection with interactive querying. It receives the detected celebrity’s name and a user question, communicates with the LLM API, and provides a concise, accurate response. This enables the application to deliver a fully interactive experience, allowing users not only to identify celebrities but also to learn more about them through natural language queries.

**6. Flask Backend Routes Code**

In this video, we will create routes for our Flask backend. These routes will handle image uploads, celebrity detection, and question answering.

We will implement this in app/routes.py.

Step 1: Import Libraries
from flask import Blueprint, render_template, request
import base64

# Import our utility modules
from app.utils.image_handler import process_image
from app.utils.celebrity_detector import CelebrityDetector
from app.utils.QA_engine import QAEngine


Explanation:

Blueprint → Organizes backend routes.

render_template → Renders HTML pages.

request → Handles HTTP GET/POST requests.

base64 → Encodes images for display on HTML pages.

process_image, CelebrityDetector, QAEngine → Utilities from previous videos.

Step 2: Create Blueprint and Initialize Classes
main = Blueprint("main", __name__)  # Create Flask blueprint

# Initialize utility classes
celebrity_detector = CelebrityDetector()
qa_engine = QAEngine()


main → Blueprint to organize routes.

Instances of CelebrityDetector and QAEngine allow us to call their methods in routes.

Step 3: Define Home Route
@main.route("/", methods=["GET", "POST"])
def index():
    player_info = None
    result_image_data = None
    user_question = None
    answer = None
    # Handle form submission
    if request.method == "POST":
        # Check if an image was uploaded
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file:  # Ensure valid image
                # Process image and get bytes and face coordinates
                image_bytes, face_box = process_image(image_file)
                # Detect celebrity
                player_info, player_name = celebrity_detector.identify(image_bytes)
                if face_box is not None:
                    # Convert image to Base64 for HTML display
                    result_image_data = base64.b64encode(image_bytes).decode("utf-8")
                else:
                    player_info = "No face detected. Please try another image."
        # Handle user question about celebrity
        elif "question" in request.form:
            user_question = request.form.get("question")
            player_name = request.form.get("player_name")
            player_info = request.form.get("player_info")
            result_image_data = request.form.get("result_image_data")
            # Ask question to QA engine
            answer = qa_engine.ask_about_celebrity(player_name, user_question)
    # Render HTML template with results
    return render_template(
        "index.html",
        player_info=player_info,
        result_image_data=result_image_data,
        user_question=user_question,
        answer=answer
    )

Step 4: Key Points

Blueprint → main organizes routes.

Route Methods → GET → fetch inputs, POST → send data from form.

Variables:

player_info → Celebrity info (name, nationality, achievements).

result_image_data → Base64 image with face box.

user_question → Question user asked about celebrity.

answer → Answer generated by QA engine.

Image Handling → Check valid image → process → detect celebrity → encode image.

Question Handling → Retrieve question, player info, image from form → send to QA engine → get answer.

Render Template → Send all variables to index.html for display.

Step 5: Flow Summary

User uploads image → POST request → validate image → process image → detect celebrity → encode image → send info to HTML.

User asks a question → POST request → extract question and previous context → QA engine returns answer → send to HTML.

GET request → Display empty form when page loads.

In short:

This blueprint handles both image upload & celebrity detection and question answering. The results (celebrity info, image with face box, user question, and answer) are sent to the HTML template for display.

Next, we will create the HTML frontend (index.html) to show all these results.

**Summary:**

In this video, we implemented the Flask backend routes for our Celebrity Detector and Question Answer System. These routes, defined in app/routes.py, handle both image uploads for celebrity detection and user questions for the QA engine. We began by importing necessary libraries, including Flask modules (Blueprint, render_template, request) for route management, base64 for encoding images to display on HTML pages, and our utility modules (process_image, CelebrityDetector, and QAEngine) from previous steps.

We created a Flask blueprint named main to organize our routes and initialized instances of CelebrityDetector and QAEngine to utilize their methods within the routes. The primary route / handles both GET and POST requests. For POST requests, if a user uploads an image, the image is processed via the process_image utility, the celebrity is detected using the CelebrityDetector, and the resulting image is converted to Base64 for rendering on the frontend. If a user submits a question, the QA engine is called with the celebrity’s name and the question to generate an answer. GET requests simply render an empty form when the page loads.

Key variables include player_info (celebrity details), result_image_data (Base64 image with detected face bounding box), user_question (question submitted by the user), and answer (response from the QA engine). The route validates the image, processes it, detects the celebrity, encodes the image, and sends all results to the HTML template (index.html) for display.

In summary, this blueprint effectively connects the backend utilities with the frontend, enabling users to upload an image, detect the celebrity, and ask questions about them. All results, including the celebrity information, processed image, user question, and generated answer, are rendered on the web page for an interactive experience.

**7. Main Application Code using HTML/CSS and Flask**

In this video, we are integrating the backend (routes/blueprint) with the frontend (HTML & CSS) to create the complete Flask application.

Step 1: Setting Up the Flask App

File: app/__init__.py

Import necessary libraries:

Flask for the web app.

os for handling paths.

load_dotenv to load environment variables.

Your main blueprint from app/routes.

Define a function to create the app:

from flask import Flask
from app.routes import main
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()  # Load environment variables
    # Template folder path
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    # Initialize Flask app
    app = Flask(__name__, template_folder=template_path)
    # Set secret key for sessions & forms
    app.secret_key = os.getenv("SECRET_KEY", "default_secret")
    # Register blueprint
    app.register_blueprint(main)
    return app


Purpose:

The secret key secures forms and session data.

Templates folder is specified so Flask knows where to find HTML files.

Blueprint (main) contains all backend routes.

Step 2: Running the Flask App

File: app.py (root directory)

Import the app creation function:

from app import create_app
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()  # Optional but ensures environment variables are loaded
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


Explanation:

0.0.0.0 makes the app accessible externally (needed for Docker/K8s deployment).

debug=True helps in development by showing errors.

Step 3: Frontend Setup

Templates folder (templates/index.html)

Two forms:

Image Upload: Upload a celebrity image.

Question Form: Ask a question about the detected celebrity.

Jinja2 Templates:

Used to bridge Python data with HTML.

Example:

<img src="data:image/jpeg;base64,{{ result_image_data }}" />
<ul>
  {% for key, value in player_info.items() %}
    <li>{{ key }}: {{ value }}</li>
  {% endfor %}
</ul>
<p>User Question: {{ user_question }}</p>
<p>Answer: {{ answer }}</p>


Notes:

Form input names (image, question) must match route handling.

Base64 encoding is used to display processed images in HTML.

Step 4: Styling

Static folder (static/style.css)

Contains all frontend styling.

Connected to HTML using <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">.

Step 5: Application Flow

User uploads an image → POST request handled in blueprint.

Image is processed → celebrity detected → bounding box drawn.

Result image, celebrity info (player_info), question, and answer are rendered in HTML using Jinja2.

User can ask questions about the celebrity → answered using the question-answer engine.

Step 6: Demo

Uploaded images of celebrities like MS Dhoni, Salman Khan, Robert Downey Jr.

Results:

Bounding box around the face.

Celebrity info displayed (name, profession, nationality, achievements).

Question-answer works perfectly.

Summary

Backend (routes) + Frontend (index.html, style.css) integrated.

Blueprint main handles all routes.

Base64 is used to render images on HTML.

Two forms: image upload and question form.

Secret key secures sessions for form submissions.

Application is production-ready and can be deployed on servers.

**Summary:**

In this video, we integrated the Flask backend routes with the frontend HTML and CSS to create a fully functional Celebrity Detector and Question Answer System. We began by setting up the Flask application in app/__init__.py, where we imported Flask, os, load_dotenv for environment variables, and our main blueprint from app/routes. We defined a create_app function that initialized the Flask app, specified the templates folder, set a secret key for session security, and registered the blueprint containing all backend routes. This setup ensures the backend is properly connected and secure, while Flask knows where to locate HTML templates.

Next, we configured the app runner in app.py in the root directory. Here, the app is created by calling create_app(), and it runs on host 0.0.0.0 and port 5000 with debug=True. The host configuration allows external access, which is essential for Docker and Kubernetes deployment, while debug mode aids development by showing runtime errors.

On the frontend, we set up the templates/index.html file with two forms: one for image upload and another for user questions. Using Jinja2 templating, we dynamically rendered Python data on the HTML page, such as the processed celebrity image (displayed via Base64 encoding) and celebrity information (player_info). Question and answer pairs from the QA engine are also displayed, ensuring an interactive interface. The frontend styling was added through static/style.css, linked in HTML to provide a polished look for the forms and results.

The application flow is as follows: the user uploads a celebrity image, which is processed by the backend; the celebrity is detected, and a bounding box is drawn on the face. The resulting image, celebrity details, and any user questions are then displayed on the HTML page. Users can ask questions about the detected celebrity, and the QA engine returns answers in real time. During the demo, celebrity images like MS Dhoni, Salman Khan, and Robert Downey Jr. were uploaded, and the application successfully displayed face bounding boxes, detailed information, and answered user questions accurately.

In summary, the Flask backend and frontend were fully integrated: the blueprint main handles all routes, Base64 is used to render images, forms for image upload and questions work seamlessly, the secret key secures sessions, and the system is now production-ready for deployment.

**8. Dockerfile , Kubernetes Deployment File and Code Versioning using GitHub**

In this video, we are containerizing the Flask app using Docker, deploying it with Kubernetes, and doing code versioning with GitHub.

Step 1: Docker Setup

File: Dockerfile (root directory)

Base Image:

FROM python:3.11


Using Python 3.11 as the parent image.

Environment Variables: (Optional but recommended)

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


Working Directory:

WORKDIR /app


Install System Dependencies (needed by OpenCV):

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6


Copy App Content:

COPY . /app


Install Python Packages:

RUN pip install --no-cache-dir -e .


Installs all packages listed in setup.py and requirements.txt without using cached files.

Expose Port & Run App:

EXPOSE 5000
CMD ["python", "app.py"]


Step 1: Summary:

Python base image

Environment variables

OpenCV dependencies

Copy all files

Install Python packages

Expose Flask port 5000

Run the app

Step 2: Kubernetes Deployment

File: kubernetes_deploy.yaml (root directory)

Deployment Section:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: lm-ops-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: lm-ops-app
  template:
    metadata:
      labels:
        app: lm-ops-app
    spec:
      containers:
        - name: lm-ops-app
          image: <GCP_ARTIFACT_REGISTRY_PATH>/lm-ops-app:latest
          ports:
            - containerPort: 5000
          env:
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: lm-ops-secret
                  key: api-key


Service Section:

apiVersion: v1
kind: Service
metadata:
  name: lm-ops-app-service
spec:
  selector:
    app: lm-ops-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer


Step 2: Notes:

Deployment name and container name should match (lm-ops-app).

replicas: 1 for simplicity; can increase later.

Port 5000 is exposed internally; service maps port 80 externally.

Service type LoadBalancer to make app public.

API keys can be injected into Kubernetes secrets and accessed from the container.

Step 3: GitHub Versioning

Create .gitignore file:

Ignore sensitive files like .env and project management files.

Initialize Local Repository:

git init


Add Remote Repository:

git remote add origin <GITHUB_REPO_URL>


Add Files & Commit:

git add .
git commit -m "Initial commit"


Push to GitHub:

git push origin main

Step 3: Notes:

Only files not listed in .gitignore will be pushed.

Check GitHub to confirm successful upload.

Step 4: Workflow Summary

Docker: Containerizes Flask app with all dependencies.

Kubernetes: Deploys app using Deployment + Service; API keys injected via secrets.

GitHub: Version control to keep track of code changes.

This setup ensures:

Reproducible environments (Docker)

Scalable deployment (Kubernetes)

Code safety and collaboration (GitHub)

**Summary:**

In this video, we focused on containerizing the Flask application, deploying it with Kubernetes, and implementing code versioning using GitHub. First, we created a Dockerfile in the project root to containerize the Flask app. We used python:3.11 as the base image and set environment variables to prevent Python from writing .pyc files and to enable unbuffered output. We set the working directory to /app and installed system dependencies required for OpenCV. The entire project content was copied into the container, and Python packages were installed using pip install -e . to include all dependencies listed in setup.py and requirements.txt. Finally, port 5000 was exposed, and the Flask app was set to run via CMD ["python", "app.py"]. This ensures that the container is fully self-sufficient and ready to run the application in any environment.

Next, we defined a Kubernetes deployment in kubernetes_deploy.yaml. The deployment specifies one replica of the containerized app with the name lm-ops-app, exposing port 5000 internally. API keys are injected securely using Kubernetes secrets, allowing the container to access necessary credentials without exposing them in code. The service section maps the internal container port to port 80 externally and uses LoadBalancer type to make the application publicly accessible. This setup allows scalable and secure deployment of the application on Kubernetes clusters, while keeping sensitive data protected via secrets.

For code versioning, we created a .gitignore file to exclude sensitive files like .env. A local Git repository was initialized, the remote GitHub repository was added, and all files were committed and pushed to GitHub. This ensures that the codebase is safely stored in version control, enabling collaboration, tracking changes, and easy rollback if needed. Only the files not listed in .gitignore are pushed to GitHub, keeping secrets and unnecessary files private.

Summary of Workflow: Docker containerizes the Flask application with all dependencies, Kubernetes handles scalable deployment and secret management, and GitHub maintains version control. This integrated setup ensures reproducible environments, secure and scalable deployments, and efficient code collaboration.

**9. GCP Setup ( Service Accounts , GKE, GAR )**

In this video, we set up Google Cloud for our project, including enabling APIs, creating a Kubernetes cluster, an artifact registry, and a service account.

Step 1: Prerequisites

Before starting, ensure you have completed:

Docker file – already created.

Kubernetes deployment file – already created.

Code versioning on GitHub – already completed.

Step 2: Google Cloud Account

Sign in to Google Cloud Platform (GCP)

New users get $300 free credits.

Step 3: Enable Required APIs

Go to APIs & Services → Library and enable the following APIs:

Kubernetes Engine API – required to create Kubernetes clusters.

Google Container Registry API – needed to store Docker images in Artifact Registry.

Compute Engine API – manages computation resources for VM instances.

Cloud Build API – required for building and deploying images.

Cloud Storage API – used for storing objects/files.

IAM API – to manage roles and service accounts.

(Optional but recommended to avoid errors in deployment)

Step 4: Create Google Kubernetes Engine (GKE) Cluster

Go to Kubernetes Engine → Clusters → Create Cluster.

Configure:

Name: lm-ops

Region: us-central1

Tier: Standard

Skip Fleet Registration.

Networking:

Access via DNS

IPv4 enabled

Do not enable authorized networks

Advanced options: leave default.

Click Create → cluster creation may take ~5 minutes.

Step 5: Create Google Artifact Registry

Go to Artifact Registry → Repositories → Create Repository

Configure:

Name: lm-ops-repo

Format: Docker

Tier: Standard

Region: us-central1

Click Create → repository is created instantly.

Step 6: Create Service Account

Go to IAM & Admin → Service Accounts → Create Service Account

Name: e.g., celebrity

Assign Permissions:

Owner

Storage Object Admin

Storage Object Viewer

Artifact Registry Admin

Artifact Registry Writer

Click Done

Step 7: Create Service Account Key

Go to Actions → Manage Keys → Add Key → Create New Key

Format: JSON → click Create → download the key.

Place the JSON file in your project root and rename it to:

GCP_key.json


Add to .gitignore to prevent pushing it to GitHub:

GCP_key.json

Summary of Google Cloud Setup

Enabled six essential APIs.

Created Kubernetes cluster (lm-ops).

Created Artifact Registry (lm-ops-repo).

Created Service Account with required permissions.

Generated JSON key for service account access and added it to .gitignore.

Note: Only move forward to the next step once the Kubernetes cluster shows a green tick (fully created).

**Summary:**

In this video, we focused on setting up Google Cloud Platform (GCP) to prepare the environment for deploying our Celebrity Detector and QA System. We started by ensuring prerequisites were complete, including having the Dockerfile, Kubernetes deployment file, and code versioning on GitHub ready.

Next, we signed in to Google Cloud Platform and noted that new users receive $300 in free credits. We then enabled the essential APIs required for the project, including Kubernetes Engine API for cluster creation, Google Container Registry API for storing Docker images, Compute Engine API for managing VM instances, Cloud Build API for building and deploying images, Cloud Storage API for file storage, and IAM API to manage roles and service accounts. Enabling these APIs ensures that all necessary GCP services work seamlessly during deployment.

After enabling APIs, we created a Google Kubernetes Engine (GKE) cluster named lm-ops in the us-central1 region with the standard tier. We used default networking settings, enabled IPv4 access, and skipped fleet registration. Cluster creation took a few minutes, and a green tick confirmed that the cluster was ready. We then created a Google Artifact Registry named lm-ops-repo with Docker format in the same region, which serves as a secure storage location for our containerized Docker images.

Finally, we set up a Service Account to manage permissions and automate deployments. The account, named celebrity, was assigned roles including Owner, Storage Object Admin/Viewer, and Artifact Registry Admin/Writer. A JSON key was generated for this account, downloaded, renamed to GCP_key.json, and added to .gitignore to keep it secure. This key allows our project to authenticate with GCP programmatically during Docker pushes and Kubernetes deployments.

Summary: The GCP setup included enabling six critical APIs, creating a Kubernetes cluster (lm-ops), creating an Artifact Registry (lm-ops-repo), creating a Service Account with necessary permissions, and generating a secure JSON key for authentication. With this setup complete and the cluster fully active, the environment is now ready for deploying our containerized application.

**10. Circle CI Pipeline Code**

In this video, we create a CircleCI pipeline to automate Docker image building, pushing to Google Artifact Registry, and deploying to Google Kubernetes Engine (GKE).

Step 1: Folder & File Setup

In the root directory, create a folder:

.circleci


Inside .circleci, create a file:

config.yml


Copy the config.yml from the GitHub repository (provided).

CircleCI will automatically detect this folder and file.

Step 2: Config.yml Overview

Version: 2.1 – specifies the CircleCI version.

Executor: Docker executor using Google Cloud SDK image (gcloud)

Why? gcloud is pre-installed; no need for manual installation.

Working Directory: /repo – all code runs here.

Step 3: Jobs in the Pipeline
Job 1: Checkout Code

Pulls your project code from GitHub into the Docker container.

Ensures the pipeline has the latest version of the code.

Job 2: Build Docker Image

Uses remote Docker to enable Docker commands inside CircleCI.

Authentication with GCP:

Fetch GCP_KEY from CircleCI environment variables (stored as base64).

Decode back to GCP_key.json.

Activate the service account using the JSON key.

Authenticate Docker with Google Artifact Registry.

Build & Push Image:

docker build -t <artifact-registry-path>/<repo-name>:<image-tag> .
docker push <artifact-registry-path>/<repo-name>:<image-tag>


Environment Variables:

GCP_KEY – base64 encoded GCP service account key.

PROJECT_ID – GCP project ID.

REGION – region for Artifact Registry and GKE.

Job 3: Deploy to GKE

Checkout code again.

Setup remote Docker.

Authenticate with GCP using the same service account key.

Deploy using Kubernetes:

kubectl apply -f Kubernetes_deployment.yaml
kubectl rollout restart deployment <app-name>


Notes:

Deployment name and image must match Kubernetes deployment YAML.

Project ID and region are pulled from CircleCI environment variables.

Step 4: Workflow Definition

Defines job execution order:

checkout code → must succeed first.

build Docker image → runs only if checkout succeeds.

deploy to GKE → runs only if Docker image build succeeds.

Ensures CI/CD pipeline executes in the correct order.

Step 5: Key Points

Service account key is never exposed publicly; stored as CircleCI environment variable.

Image names and deployment names must be consistent across CircleCI config and Kubernetes YAML.

Pipeline automates:

Code checkout

Docker image build & push

Kubernetes deployment

Next Steps:

In the following video, environment variables in CircleCI will be set.

The pipeline will be executed to deploy the app to Google Kubernetes Engine.

**Summary:**

In this video, we set up a CircleCI pipeline to automate the deployment workflow for our Celebrity Detector and QA System. The pipeline is designed to build the Docker image, push it to Google Artifact Registry (GAR), and deploy it to Google Kubernetes Engine (GKE) automatically, reducing manual intervention and ensuring consistent deployments.

We started by creating a .circleci folder in the project root and adding a config.yml file inside it. This configuration file, which CircleCI automatically detects, contains the pipeline instructions. We used CircleCI version 2.1 and a Docker executor with the Google Cloud SDK image to ensure that gcloud commands are available without manual installation. The working directory inside the executor is set to /repo, where all project code is executed.

The pipeline consists of three main jobs. The first job, Checkout Code, pulls the latest code from GitHub into the Docker container to ensure the pipeline works with the most recent changes. The second job, Build Docker Image, sets up remote Docker so that Docker commands can run inside CircleCI, decodes the base64-encoded GCP service account key, activates the service account, authenticates Docker with GAR, and then builds and pushes the Docker image. This job relies on environment variables such as GCP_KEY, PROJECT_ID, and REGION to manage authentication and deployment paths securely.

The third job, Deploy to GKE, again checks out the code, authenticates with GCP using the service account key, and applies the Kubernetes deployment using kubectl apply. It also performs a rollout restart to ensure the deployment picks up the new image. The job depends on the Docker image being successfully built and pushed to GAR. The workflow section in config.yml ensures that jobs run in the correct order: first checkout, then build, then deploy.

Key points include keeping the service account key secure by storing it as a CircleCI environment variable, maintaining consistency in image and deployment names across CircleCI config and Kubernetes YAML, and automating the entire CI/CD process. This setup ensures that whenever code is pushed to GitHub, CircleCI can automatically build the Docker image, push it to GAR, and deploy it to GKE without manual steps.

Summary: The CircleCI pipeline automates the workflow from code checkout to Docker image build and push, and finally to Kubernetes deployment, ensuring reproducible, secure, and efficient CI/CD for the application.

**11. Full CI/CD Deployment of Application on GKE**

In this video, we perform the CircleCI setup, configure environment variables, deploy the application to GKE, and perform cleanup.

Step 1: Convert GCP Key to Base64

Open Git Bash in VSCode.

Run the command to encode your GCP_key.json:

base64 GCP_key.json


Copy the encoded output (ensure no extra spaces at start or end).

Step 2: CircleCI Project Setup

Sign up/login to CircleCI
.

Create a new project:

Name: LM_ops (or anything you prefer).

Pipeline name: Build and Test.

Connect GitHub repository (authorize if required).

Push the .circleci/config.yml to GitHub:

git add .
git commit -m "Add CircleCI config"
git push


CircleCI will auto-detect config.yml.

Set triggers:

Default: run on all pushes.

Optional: run only on specific branches/events.

Step 3: Define Environment Variables

In Project Settings → Environment Variables, define the following:

Name	Value
GCLOUD_SERVICE_KEY	Base64 encoded GCP key JSON
PROJECT_ID	GCP Project ID
GKE_CLUSTER	Name of your Kubernetes cluster
COMPUTE_REGION	Region where GKE cluster is deployed

Ensure no extra spaces when pasting the base64 key.

Step 4: Running the Pipeline

Automatic: triggers on push to GitHub.

Manual:

Go to Pipelines → LM_ops project.

Select branch (e.g., main) and pipeline (Build and Test).

Click Run Pipeline.

Pipeline stages:

Checkout code

Build Docker image

Deploy to GKE

Step 5: Kubernetes Secrets

If your deployment requires secrets (like API keys):

kubectl create secret generic <secret-name> --from-literal=API_KEY=<your-api-key>


Example: inject Grok API key into your cluster.

Resolves errors like container configuration error.

Step 6: Verify Deployment

Go to GKE → Workloads → LM_ops app.

Wait 5–10 minutes if you see errors like minimum availability, especially on trial accounts.

Open the Load Balancer endpoint to access your app.

Test the functionality (e.g., celebrity detection demo).

Step 7: Cleanup (Important!)

To avoid unnecessary charges:

Delete Kubernetes cluster:

GKE → Clusters → Delete cluster.

Delete Docker Artifact Repository:

Artifact Registry → Delete repository.

Optional:

Delete CircleCI project

Delete service account (not charged for service accounts)

Only cluster and artifact repository deletion are mandatory to avoid costs.

Key Takeaways

Encode GCP keys in base64 to store safely in CircleCI environment variables.

CircleCI automates the full CI/CD pipeline: code checkout → Docker build → push → GKE deployment.

Kubernetes secrets must be injected to avoid container errors.

Always perform cleanup to avoid unnecessary GCP charges.

**Summary:**

In this video, we completed the CircleCI setup, configured environment variables, deployed the application to GKE, and performed cleanup to ensure a smooth and cost-efficient CI/CD workflow.

The first step involved converting the GCP service account key (GCP_key.json) to base64 using Git Bash in VSCode. This ensures the key can be securely stored in CircleCI environment variables without exposing sensitive data. The encoded key is copied carefully to avoid extra spaces, which could cause authentication errors.

Next, we set up the CircleCI project by signing in, creating a new project (e.g., LM_ops), connecting it to our GitHub repository, and pushing the .circleci/config.yml file. CircleCI automatically detects this configuration and can trigger pipelines on every push to GitHub or manually when needed.

We then defined essential environment variables in CircleCI Project Settings, including GCLOUD_SERVICE_KEY (base64-encoded key), PROJECT_ID, GKE_CLUSTER, and COMPUTE_REGION. These variables allow the pipeline to authenticate with GCP, locate the correct cluster, and deploy the application automatically.

Running the pipeline triggers the stages sequentially: code checkout → Docker image build → push → deployment to GKE. For deployments that require secrets, such as the Grok API key, we used Kubernetes secrets via kubectl create secret, ensuring the container runs without configuration errors.

After deployment, we verified the application on GKE workloads and accessed it via the Load Balancer endpoint to confirm that features like celebrity detection and question answering work as expected.

Finally, we performed cleanup to avoid unnecessary charges. This involved deleting the Kubernetes cluster and Artifact Registry repository. Optionally, CircleCI projects and service accounts can also be removed.

Key Takeaways:

Encode GCP keys in base64 to store safely in CircleCI.

CircleCI automates the full CI/CD process: code checkout → Docker build → push → GKE deployment.

Kubernetes secrets must be injected to avoid container configuration errors.

Always clean up cloud resources like clusters and repositories to prevent unexpected costs.

This setup ensures a secure, automated, and cost-efficient deployment workflow for the Celebrity Detector and QA system.

**F. AI Music Composer**

**Tech Stack:**

1. Groq - LLM

2. Gitlab CI/CD - CI/CD Tool for making Pipelines

3. Langchain - Gen AI Framework to interact with LLM

4. Gitlab - Serves as SCM for our Project

5. Music21 - To handle music theories like notes, chords, etc..

6. Streamlit - To make frontend of the app

7. Docker - For Containerization off app during deployment

8. Synthesizer - For generating sound waves (E.g. Sine waves) from notes

LLM generates Music Notes (Text Format); We will convert music notes to actual music notes and chords using **Music21**; We will convert it to actual music using **Synthesizer**

9. GCP GAR - For Storing Docker Images

### GAR - Google Artifact Registry - It is like Docker Hub for Google

10. GCP GKE - To deploy and run app on cloud within a Kubernetes Cluster

### Project Notes:

In the first stage of the project, we begin with the basic setup of the environment and API integration. The process starts by creating a virtual environment to keep all dependencies isolated and manageable. Once the environment is ready, we implement logging and custom exception handling so that any errors can be properly tracked and debugged. After this, we install all the required libraries for the project, including LangChain, Music21, and Synthesizer, which will play a crucial role later. As part of this setup, we also connect our project with the cloud by generating an API key, which allows us to use the LLaMA 3.3 language model in our application. This completes the API setup stage.

The second stage of the project involves building the utility functions that will help in processing musical data. Since the LLM will generate musical notes and chords, we need to transform them into formats suitable for audio synthesis. For this purpose, we create two helper functions. The first function uses Music21 to convert note names into frequencies, as the synthesizer can only process numerical frequency values. The second function uses the Synthesizer library to generate WAV audio files from these frequencies. WAV is chosen over MP3 since it provides uncompressed, higher-quality sound. Together, these functions bridge the gap between symbolic music representation and actual sound output.

Once the utilities are ready, we move to the core logic of the application, which is implemented inside a dedicated Music class. This class contains several methods responsible for generating different aspects of music. It first generates a melody, which is essentially a sequence of musical notes. Then, it creates harmony chords, which enhance the melody and make it richer. Next, it defines the rhythm, which refers to the beats or tempo of the composition. The system also adapts the output to different music styles, such as jazz, sad, happy, or joyful, based on user input. Finally, the class produces a composition summary that describes the generated melody, chords, rhythm, and overall style, ensuring transparency of how the music was created.

After the core music logic is complete, the fourth stage focuses on building the user interface. We use Streamlit to create a simple web application where users can type in their preferences (for example, “happy piano music” or “sad violin music”) and generate music at the click of a button. The Streamlit app handles all inputs and displays the outputs in an interactive and user-friendly manner.

The next stage is containerization of the application. We create a Dockerfile that defines how the application should be packaged into a Docker image. This ensures consistency across environments and makes deployment easier. Once the Docker image is ready, we move to Kubernetes deployment, where a deployment file specifies details like the number of replicas, ports, and the source of the Docker image. The application is then deployed to Google Kubernetes Engine (GKE) for scalable cloud deployment.

For source code management, we use GitLab instead of GitHub. All project code is pushed and maintained in a GitLab repository. Finally, the project integrates GitLab CI/CD pipelines to automate the deployment process. The pipeline is defined in a GitLab CI/CD file with three stages: the first handles code checkout (although in GitLab this is implicit), the second builds and pushes the Docker image to an artifact registry, and the third deploys the latest image to GKE. This setup ensures continuous integration and continuous deployment. Any changes in the codebase, such as updates to the UI or modifications in the logic, are automatically detected by the pipeline, and the application is redeployed with the latest updates.

By the end of this workflow, the project achieves a fully automated AI-powered music generation system that takes user input, generates musical notes using a language model, processes them into frequencies, synthesizes them into audio files, and finally serves the results through a scalable web application deployed on the cloud.

**(i) Project and API Setup (Groq)**

To begin, open any directory of your choice and create a new folder for the project. Let’s name this directory iMusic Composer. Once the directory is created, open it in VS Code. You can do this in two ways: either directly right-click and select Open with Code, or open the terminal inside that folder and type code . which will automatically launch VS Code. After opening, make sure to trust the authors when prompted. On the top menu, open the terminal and create a new Command Prompt terminal.

The first step is to create a virtual environment. For this, type the command python -m venv env (you can name the environment anything; here we use “env”). It may take some time to create. While it is being created, also make two new files: requirements.txt and setup.py. The setup.py file has already been explained earlier in the course and is also available in the GitHub resources section. You can copy the base code from there and simply modify it by changing the project name to something like music composer. To summarize, the setup.py file serves two purposes: it installs all the packages listed in the requirements file and also treats specific folders as packages rather than ordinary folders. For example, if we create a source directory, by default it is just a folder. But with the help of setup.py and an __init__.py file, it can be treated as a package, allowing imports across different parts of the project. This makes package and import management easier.

Once the environment is created, activate it by running the command env\Scripts\activate. If activated correctly, you will see (venv) in brackets on your terminal prompt. Clear the screen with cls for better readability. Next, list all the project requirements inside requirements.txt. The libraries needed are: Streamlit (for building the user interface), Music21 (for handling chords and notes in music), python-dotenv (for loading environment variables such as API keys), LangChain (the generative AI framework we will use), LangChain-core, LangChain-community, and LangChain-groq (for interactions with the Groq API). Additionally, we include scipy (which will be useful later) and the Synthesizer library (for creating and synthesizing music). Be careful while writing the spelling of “synthesizer” correctly as s y n t h e s i z e r. Once done, save the requirements.txt file.

Now, let’s define the project structure. Create a folder named app which will contain the main components of the application such as utility functions, the main logic, and eventually the application code. To ensure this folder is treated as a package, add an __init__.py file inside it. With this, our basic project structure is ready. Later, we will extend it with files like application.py, a Dockerfile, and Kubernetes deployment files, but for now, the structure is kept simple.

To install both the requirements and the package, we use the setup.py file. Run the command pip install -e . which will trigger the setup file and install everything accordingly. Once the installation is complete, we move on to API integration. Open your browser and search for Groq Cloud API. Sign up or log in, and then navigate to the API Keys section. Create a new API key, naming it something like music, and submit it. The system will generate an API key for you. Copy this key, return to your project root directory, and create a .env file. Inside this file, create an environment variable named API_KEY and assign your copied key to it, enclosing it in double quotes. This will ensure that your sensitive key is safely loaded from the environment instead of being hard-coded.

So, to summarize what we achieved in this video: we first created a virtual environment, defined the initial project structure, listed all required libraries in requirements.txt, and installed them using setup.py. We then generated and stored an API key from Groq Cloud by placing it into a .env file as an environment variable. This completes the basic setup phase. In the later stages, we will add more files for application logic, Dockerization, and Kubernetes deployment, but this foundational setup ensures the project is ready to move forward.

**1. Creating a Utility Function:**

In this part of the project, the focus was on building utility functions that would form the backbone of music generation. Inside the app directory, a new file named utils.py was created. This file serves as a dedicated module where helper functions for handling music data and generating sound are defined. To begin, five key libraries were imported.

First, Music21 was imported to handle music theory concepts such as notes, chords, and durations. This library is essential because it allows the program to interpret symbolic music elements and translate them into computational representations. Next, NumPy was included since audio signals can be represented as arrays, and NumPy provides the necessary mathematical operations to handle these signals efficiently. The IO module was also imported to enable in-memory storage of generated audio files. Instead of saving every generated file permanently on disk, temporary memory is used to store audio files during a session. This prevents storage overflow when multiple users generate files, as the cached data is cleared when the app refreshes.


**2. Core Code for Application:**

In this stage, we begin writing the core application code that ties together the utility functions we built earlier with the LLM (Large Language Model). For this, a new file named main.py is created inside the application directory. The goal of this module is to define a reusable class that manages prompts, communicates with the Groq API, and generates musical note sequences based on user input.

The first step is to perform the necessary imports. We import os to securely fetch environment variables, such as the API key required for Groq. Next, we bring in ChatPromptTemplate from LangChain, which is used to structure the prompts we send to the LLM. Finally, we import ChatGroq from LangChain’s Groq integration. This will serve as the interface for initializing and running our chosen LLaMA model via Groq Cloud.

With the imports complete, we define a class named MusicLM. This class encapsulates all the functionality related to music generation. In the constructor (__init__), we accept a parameter called temperature, with a default value of 0.7. The temperature parameter plays a crucial role in controlling the creativity of the model. At lower values (close to 0), the model behaves more rigidly and adheres strictly to factual content. At higher values (close to 1), the model generates more creative, diverse, and sometimes unexpected outputs. Since music composition requires creativity, a temperature around 0.7–0.8 is ideal, striking a balance between variety and coherence.

Inside the constructor, the LLM instance is initialized using ChatGroq. Three key arguments are provided:

Temperature – the creativity setting explained above.

API key – securely fetched using os.getenv().

Model name – one of Groq’s available models. Groq offers both lightweight and larger parameter models. For instance, llama-3.1-8b-instant (8 billion parameters) is faster and suited for quick responses, whereas llama-3.3-70b-versatile (70 billion parameters) generates more detailed and nuanced results. For this project, the smaller 8B instant model is selected to keep inference times manageable.

Next, we define the generate_melody method. This method accepts user input (a text description such as “make a happy melody”) and produces a melody—essentially a sequence of musical notes. To achieve this, a prompt template is constructed using ChatPromptTemplate.from_template(). The template instructs the LLM to generate a melody formatted as a space-separated sequence of notes.
The next import was from scipy.io.wavfile, specifically the write function. This module is responsible for converting NumPy arrays into actual WAV audio files. Just as videos come in multiple formats (e.g., MP4, MKV), audio files also have formats such as MP3 or WAV. WAV is an uncompressed audio format, while MP3 is its compressed version. In this project, WAV files are generated first, which can later be compressed into MP3 if needed. Finally, from the Synthesizer package, both the Synthesizer and Waveform classes were imported. Music21 helps generate notes, but the synthesizer converts those notes into actual sound waves — typically sine waves or similar — that form the basis of audible music.

With the imports complete, the first function was defined: note_to_frequencies(note_list). This function takes in a list of musical notes (for example, “C4” or “E4”) and converts them into corresponding frequencies measured in Hertz. A blank list named freqs was initialized to store the resulting values. The function iterates through each note in the input list, creating a Note object using Music21, then accessing its pitch to extract the exact frequency. Each valid frequency is appended to the list. Importantly, this process is wrapped in a try–except block. The reason is that users may sometimes provide invalid note names. Without exception handling, such errors would crash the program. By using except: continue, invalid entries are skipped gracefully while valid notes are still processed. Finally, the function returns the full list of frequencies, ensuring that only clean, usable data moves forward in the pipeline.

The second function was named generate_wav_bytes_from_notes(note_frequencies). This function takes the list of frequencies generated by the first function and produces actual WAV audio data. Using the Synthesizer and Waveform classes, sound waves are synthesized based on the given frequencies. These waves are then written into a WAV format using the scipy.io.wavfile.write method. Instead of saving the file directly to disk, the audio is stored in an in-memory buffer provided by the IO module. This ensures efficient storage management and makes it easy to return the audio directly to the user within the application. The end result is a temporary WAV file that represents the generated music, ready to be played back or streamed without consuming large amounts of permanent storage.

Together, these two utility functions complete a critical step in the workflow:

(i) Convert notes to frequencies (bridging symbolic music theory and physics of sound).

(ii) Generate WAV audio (turning frequencies into audible music through synthesis).

This modular design allows the app to handle both the interpretation of musical notes and the generation of audio, making it a flexible foundation for building more advanced features later in the project.

Once the note frequencies were obtained using the first function, the next task was to transform those frequencies into actual audio signals. While Music21 was used earlier for handling music theory concepts such as notes and chords, this stage relies on the Synthesizer library. The distinction between the two is important: Music21 interprets symbolic music queries (like “C4” or “E4”), whereas the Synthesizer generates the corresponding sound waves, turning theoretical notes into audible music.

To achieve this, a Synthesizer object was created. The variable synth was initialized with a sine wave configuration, since sine waves are considered the most beginner-friendly and fundamental sound waves. The synthesizer was set to use one oscillator only (osc1), while a second oscillator (osc2) was disabled by setting it to False. For the sine wave, the default volume was kept at 1.0, ensuring a balanced signal without distortion. Additionally, a sample rate of 44,100 Hz was defined. This is the industry standard for WAV files and ensures smooth playback quality. It is the same sample rate you often see when downloading MP3s labeled as 44 kHz, representing the number of samples captured per second in the digital audio signal.

With the synthesizer prepared, the process of audio generation began. For each frequency in the list of note frequencies, the synthesizer was instructed to generate a constant sine wave. Using NumPy’s concatenate function, these individual sine waves were then combined into a single continuous waveform, producing a complete audio signal. This step ensures that the sequence of notes is seamlessly merged into one coherent sound rather than isolated fragments. The resulting audio data was stored as a NumPy array, which served as the raw digital representation of the music.

Next, the generated NumPy audio array needed to be converted into a WAV file. Instead of saving it directly to the local disk, the audio was written into an in-memory buffer using Python’s IO module. This temporary storage solution prevents unnecessary use of disk space and allows audio to be retrieved or streamed instantly during a user’s session. The wavfile.write function from SciPy was employed for this conversion. It was provided with three arguments: the buffer (as the storage destination), the sample rate (to preserve quality standards), and the audio array (converted into float32 format to ensure compatibility). Finally, the function returned the stored data using buffer.getvalue(), making the WAV bytes accessible without permanent storage.

In summary, the second utility function bridges the gap between abstract musical frequencies and playable audio files. The workflow can be described as follows:

(i) A synthesizer object is created, configured to use a sine wave at a standard sample rate.

(ii) Each note frequency is processed into a sine wave and then merged into a full waveform using NumPy.

(iii) The final waveform is written into an in-memory buffer as a WAV file, ready to be played or delivered to the user.

By combining the first and second utility functions, the application is capable of taking text-based musical notes, converting them into frequencies, and ultimately producing an audio file in real time. This structured pipeline makes the system both efficient and scalable for generating music dynamically.

By specifying this output format, we ensure that the generated response is structured and easy to parse in subsequent steps.

After defining the prompt, it is combined with the initialized LLM into a chain. This is done by piping (|) the prompt into the model. Once the chain is created, it can be invoked using the user’s input. The method then extracts the relevant portion of the LLM’s response by accessing the .content attribute. A .strip() function is applied to remove any unnecessary whitespace at the beginning or end of the output. This ensures that the returned string is a clean list of notes.

To summarize, the generate_melody method performs the following steps:

(i) Accepts user input describing the kind of melody desired.

(ii) Builds a structured prompt that asks the LLM to generate a melody as a sequence of notes.

(iii) Forms a chain that connects the prompt with the Groq LLM.

(iv) Invokes the chain using the user input.

(v) Returns the generated melody as a clean, space-separated string of notes.

This design ensures that the application can dynamically generate music note sequences in response to any textual description. The next step will extend this approach by defining a second method, generate_harmony, which builds harmonic layers to accompany the generated melody.

Once the melody has been generated, the next step is to enhance it by adding harmony, rhythm, and stylistic variations. These layers transform a simple sequence of notes into a more complete and musical composition.

**Generating Harmony** - The method generate_harmony accepts a melody (produced earlier) and generates chords that harmonize with it. In music, harmony ensures that notes are not played in isolation but are grouped into chord structures, creating depth and fullness.

To achieve this, a new prompt template is created. Instead of asking for single notes, the prompt instructs the LLM to “create harmony chords for this melody”. The input melody is passed into this prompt, and the output is expected in a structured format where chords are represented as space-separated notes.

Here, each group of three notes (separated by spaces) forms a chord, and the sequence of chords provides harmony to the melody.

The logic is the same as in melody generation:

(i) Define a structured prompt.

(ii) Combine it with the Groq LLM into a chain.

(iii) Invoke the chain using the melody as input.

(iv) Extract and clean the response to return a chord progression.

Effectively, the melody provides the skeleton, while the harmony builds the supporting structure to make the music sound complete.

**Generating Rhythm:** The method generate_rhythm adds the concept of timing and beats to the composition. Rhythm determines how long each note or chord is played, giving the piece a sense of tempo and flow.

Similar to the earlier methods, this function uses a prompt that asks the LLM to “suggest rhythm durations in beats for this melody”. The format here differs, as the output must be numerical beat values separated by spaces

This means the first note is held for one beat, the next two for half a beat each, and the last for two beats. Higher values produce slower, more extended sounds, while smaller values create faster, energetic passages.

Again, the flow is identical:

(i) A rhythm-specific prompt is defined.

(ii) It is combined with the LLM into a chain.

(iii) The chain is invoked using the melody.

(iv) The cleaned output is returned as a sequence of beat durations.

By combining melody, harmony, and rhythm, we now have the essential building blocks of structured music.

**Adapting to Music Style:** Finally, the method adapt_style integrates user preferences to shape the overall feel of the music. This function accepts:

style (user-defined, e.g., jazz, classical, lo-fi, instrumental, etc.),

melody (generated earlier),

harmony (chord progression),

rhythm (beat structure).

The purpose of this method is to guide the LLM in merging all three elements into a composition that reflects the chosen style. For instance, a jazz style might favor complex chord progressions and syncopated rhythms, while a lo-fi style would emphasize simplicity and relaxed beats.

By layering style adaptation on top of melody, harmony, and rhythm, the model is capable of producing music that is not only structurally complete but also aligned with a specific mood or genre.

**Adapting to a Music Style:** After generating the melody, harmony, and rhythm, the final step is to adapt the composition to a particular style. While the previous methods create the raw musical elements, this stage ensures that the output matches the mood or genre specified by the user.

The adapt_style method achieves this by constructing a prompt that integrates all three elements—melody, harmony, and rhythm—along with the chosen style. The format of the prompt is simple and structured:

(i) First, it specifies the style to be applied (e.g., “Adapt to happy style” or “Adapt to jazz style”).

(ii) Then, it appends the melody sequence.

(iii) Next, it includes the harmony (the chords generated earlier).

(iv) Finally, it lists the rhythm (the beat durations).

This organized prompt is then passed to the language model chain, ensuring that all components are processed together. When invoking the chain, the system passes a dictionary containing the user-selected style, the generated melody, the harmony, and the rhythm. Each of these keys must exactly match the input parameters expected by the template to avoid mismatches. The model then produces a single string summary, which represents the adapted piece of music in the specified style. To ensure clean output, the response is stripped of extra spaces before returning.

**Why Style Adaptation is Important:** The reason for combining everything into a single string summary lies in the transformation process. By this stage, the system already has:

(i) A melody – a raw sequence of notes.

(ii) A harmony – chords that organize the melody.

(iii) A rhythm – the timing and beat structure.

Individually, these represent separate musical elements, but they are not yet cohesive. Style adaptation restructures these elements into a unified musical piece that reflects the user’s preference. For example:

(i) If the style is set to happy music, the model will arrange the melody, harmony, and rhythm in a bright, uplifting manner.

(ii) If the style is sad or sorrowful, the same elements will be restructured to sound slower, softer, and emotionally deeper.

(iii) If the style is jazz, classical, or lo-fi, the chords, rhythms, and note progressions will be aligned to match those genres.

Thus, style adaptation acts as the final layer of polish—it takes the flat, technical music structure and molds it into something emotionally and stylistically meaningful.

### Complete Workflow Summary:

To summarize the entire workflow:

(i) Constructor Setup – Initializes the LLM with API keys, temperature, and model parameters.

(ii) Generate Melody – Produces a raw sequence of musical notes (e.g., “C4 D4 E4 G4”).

(iii) Generate Harmony – Groups these notes into chords, giving them a structured arrangement.

(iv) Generate Rhythm – Assigns beats and timing, defining how long each note or chord should be played.

(v) Adapt Style – Combines melody, harmony, and rhythm into a single string, adapting the music to the user’s chosen style (e.g., happy, sad, jazz, lo-fi).

This pipeline ensures that music is generated in a step-by-step, modular fashion. Each component builds on the previous one, and the final adaptation step ensures that the composition reflects the user’s creative intent.

**3. Main Application code using Streamlit**

A new file must be created in the root directory and named app.py.

First, some essential imports are required. Streamlit is imported to create the front end of the application. Streamlit is referenced as st. The Music class from the previous video is imported using from app.main import Music. Additionally, utility functions are imported from the app/utils folder. The utils file contains two functions: notes_to_frequencies and frequencies_to_audio. Both functions are imported at once using from app.utils import *.

The BytesIO class from the io module is imported to handle audio bytes. This allows the generated music to be played directly within the Streamlit app. The load_dotenv function is imported from the dotenv module to ensure all environment variables are loaded whenever app.py runs. This is particularly important during deployment using Docker, as all environment variables will be loaded automatically, preventing issues with the Grok LM.

Next, basic configuration for Streamlit is performed. The function st.set_page_config() is called at the top of the file, as it must be executed before any other Streamlit code. The page title is set to "Music Composer", and the layout is centered. The app title is defined with st.title("Music Composer"). A description is added using st.markdown(), such as “Generate music by describing the style and contents.” Running streamlit run app.py in the terminal confirms that the title and description appear correctly.

An input section is then created to accept two inputs from the user: the type of music and the style. The music input is collected using st.text_input(), allowing the user to describe the music they wish to compose. The style is selected from predefined options using a st.selectbox(). Available styles include “Sad,” “Happy,” “Jazz,” “Romantic,” and “Extreme.” Additional styles can be added as required. Once the app is saved and refreshed, the two input boxes are visible, enabling user interaction.

A button is created using st.button() and labeled “Generate Music.” When pressed, the app verifies that the music input is not empty before generating music. An object of the MusicLM class is created using generator = MusicLM(). A spinner is added with with st.spinner("Generating music...") to provide visual feedback while the music is being generated.

The application generates four components: melody, harmony, rhythm, and the final composition. Melody is produced using generator.generate_melody(), based on the user-provided input. Harmony is generated with generator.generate_harmony(melody), which takes the generated melody as input. Rhythm is produced using generator.generate_rhythm(melody). The final composition is created using generator.adapt_style(), which combines the user-selected style, melody, harmony, and rhythm. This function provides an overall summary of how the music was composed.

To convert generated notes into frequencies, the melody string is split into a list using melody.split(). For example, a string "E4 E5 E6" becomes ['E4', 'E5', 'E6']. This list is stored as melody_notes and converted to frequencies using the notes_to_frequencies() function, with the result stored in melody_frequencies. The same process is repeated for harmony, splitting the harmony string into chords and converting them into frequencies.

This organized process ensures that all inputs, including the user-selected style, generated melody, harmony, and rhythm, are properly handled. The final composition reflects the combined result of all components, providing an accurate representation of the music as intended by the user.

The harmony string is first split into individual chords. An empty list is initialized to store the harmony notes. Each chord in the harmony is iterated over using a loop, and the notes within each chord are split and extended into the list. This process ensures that each chord is separated into individual notes, similar to the way the melody notes are handled.

For example, a chord sequence such as "C4 E4 G4" is split into ['C4', 'E4', 'G4']. This organization allows the harmonics to complement the melody, enhancing the overall musical structure. The main purpose of splitting the harmonic chords is to make the music more organized and structured.

Once the harmony notes are properly split, the frequencies for both melody and harmony are generated. The notes_to_frequencies() function is used for this purpose. Melody frequencies are obtained from the melody notes, while harmony frequencies are obtained from the harmony notes. After computing the frequencies, the melody and harmony frequencies are combined into a single list, representing all frequencies that will be used to generate the final music.

To summarize the process so far: first, the page configuration is defined; then the user inputs for music and style are set up. A button is created, and an object of the MusicLM class is initialized. Melody, harmony, rhythm, and composition are generated. The melody notes are converted to frequencies, and harmony chords are split into notes and converted to harmonic frequencies. Both frequency lists are then combined for audio generation.

The melody notes form the main tune of the music, while the harmony notes act as an improvised version that enriches and harmonizes the melody. Combining both ensures that the music is fuller and more pleasing to the listener.

The combined frequencies are then used to generate a WAV audio file using the generate_wav_bytes_from_notes_frequencies() function. The resulting audio is stored in a variable named wav_bytes. Since the audio is in bytes format, it cannot be directly displayed in the Streamlit app. To address this, the BytesIO class is used, and the audio is rendered with st.audio(). The audio format is specified as WAV; if a different format such as MP3 is desired, the format parameter can be adjusted accordingly. A success message is displayed to indicate that the music has been generated.

Additionally, the composition summary is shown using st.expander(). This allows users to expand and view the generated composition details, including the melody, harmony, rhythm, and how the music was structured. The rhythm component is primarily used in the composition generation; the synthesizer library used does not support advanced rhythm integration. Alternative libraries, such as Midi, can be used to include rhythm more explicitly.

A brief summary of the application: essential imports include Streamlit, the MusicLM class, utility functions, load_dotenv for environment variables, and BytesIO for displaying audio. The app page is configured, user inputs are collected, and a button is created. Melody, harmony, rhythm, and the overall composition are generated, converted to frequencies, and combined. The final WAV audio file is displayed using BytesIO, and the composition summary is provided.

Testing the application involves generating prompts to input into the music description box. For example, a prompt like “alien soundscape with unusual harmony and experimental rhythm” is entered, and a style is selected, such as Extreme. The generated music plays, and the composition summary is displayed, showing details of the melody, chords, and rhythm.

Other prompts can also be tested, such as “generate a relaxing jazz tune with melody and soft rhythm.” The generated output demonstrates variations in melody and harmony according to the selected style. The synthesizer library is limited to a single instrument; other libraries can be used to simulate instruments like guitar or violin. The composition summary provides additional details, including chord progressions, rhythm, tempo, and melodic characteristics.

This concludes the creation and testing of the application. The app successfully accepts user inputs, generates music using the MusicLM class, displays audio, and presents the composition summary. The process is now complete, and the application is ready for further exploration or enhancement.

**4. Docker File and Kubernetes Deployment File:**

In this video, we will be creating our Dockerfile and our Kubernetes deployment file. First, let’s make a Dockerfile. You can go to the GitHub resource section provided, where a link to the GitHub repository is given. Inside that repository, you will find a Dockerfile. Simply copy it and paste it here. Now, let me explain what this Dockerfile is doing. It is based on the Python 3.11 slim image, and then we create one working directory called app. Whenever this Dockerfile is used to build a container, that directory will be formed. Inside this app directory, all the project files will be copied—whether it is app.py, the app folder, your requirements.txt, or setup.py—everything will be placed inside this directory.

pip install -e .

In the Dockerfile, the same approach is used, but with --no-cache-dir added. This flag avoids creating cache files inside the app directory and ensures a fresh restart for the application. After that, some default environment variables are set, and finally, the command to run the application is defined. In the previous video, we ran our application using: streamlit run app.py

Now, let’s discuss the three key parameters: server port, server address, and server headless. The server port is set to 8501, because that is the default port for Streamlit applications. The server address is set to 0.0.0.0, which means the app can be accessed from anywhere, not just localhost. This is important if the application is deployed on a virtual machine or Kubernetes cluster, since the IP will change. The server.headless = true setting is also crucial. By default, Streamlit automatically tries to open the browser when you run the app locally (because headless is false). For example, if you run streamlit run app.py locally, you’ll see the browser automatically opens. But in production environments (e.g., Linux servers), you don’t want this, as it can cause dependency errors. That’s why we explicitly set headless to true—so it won’t try to force-open a browser.

That’s all about the Dockerfile. Now, in the root directory, we will create another file called kubernetes-deployment.yaml. You can again go to the GitHub repository mentioned in the resources, find the YAML file, copy it, and paste it here. A Kubernetes deployment file has two main parts: the Deployment and the Service.

The deployment name is set as lmops-app, and you must ensure that this name is used consistently across the YAML file, including in the service section. Next is replicas, which specifies how many pods of your application you want. Here it is set to 1 for simplicity, but you could set it to 2 or 3 if needed. More replicas mean higher costs, and since this is a learning project (not a real production company project), keeping it low avoids unnecessary expense.

Then comes the image path. This is the Docker image for your application, which will be stored in Google Artifact Registry. The path format includes the artifact registry, project ID, repository name, and image name. Kubernetes will fetch the image from there to create and deploy your container. Next, the container port is defined as 8501, because that’s the port where the application is running. You could also expose this port in the Dockerfile (optional), but it is not strictly necessary.

The service part of the YAML links to this deployment. Here we define the service name (lmops-service), which acts on the deployment lmops-app. The target port is again set to 8501, and it must match the container port. As for the service type, Kubernetes supports three types: ClusterIP, NodePort, and LoadBalancer. ClusterIP is used for internal deployments, while NodePort and LoadBalancer are better for external, internet-facing deployments. LoadBalancer is preferred for production-grade external deployments.

Finally, we have environment variables. When doing code versioning, we don’t push sensitive files (like .env containing API keys) to the repository. Instead, we inject the API key directly into the Kubernetes cluster as a secret. The application then fetches the API key from Kubernetes at runtime. That’s what the environment section in the YAML is doing—it loads the API key into the application securely.

So, to summarize: in this video, we created a Dockerfile and a Kubernetes deployment file. In the Dockerfile, we defined the Python base image, working directory, dependency installation, environment settings, and the run command. In the Kubernetes YAML, we set the deployment name, replicas, image path, container port, service configuration, and environment variable handling.

**5. Code Versioning using GitLab:**

In this video, you will be doing code versioning using GitLab. First, let’s create a .gitignore file. In the root directory, create a file named .gitignore. In this file, you list all the files and folders that you don’t want to push to the repository. For example, we will exclude three things: the .env file, the virtual environment, and the project management files. To do this, copy their relative paths and paste them into the .gitignore, adding a backslash where necessary. These three should not be pushed because the virtual environment contains large files, the environment file holds sensitive API keys, and the project management files are not required in version control.

Next, go to your browser, open gitlab.com, create a new account (if you don’t already have one), and log in. After logging in, create a new project. Choose blank project, then give it a name—for example, let’s call it music. You can select the deployment target as Kubernetes (though this step is optional). Make sure to set the project visibility to public. Also, do not initialize the repository with a README file. Once that is set, create the project.

After the project is created, GitLab will show you setup instructions. You need to initialize your repository locally. Make sure you are inside the HTTPS section (not SSH). Copy the first command provided (git init) and run it inside your terminal in VS Code. To do this, open a CMD terminal in VS Code, ensure your virtual environment is activated, paste the command, and press Enter. This initializes your Git repository.

Next, you need to add the origin. Copy the GitLab HTTPS URL from your repository and run the git remote add origin command. You might see it added twice, but you only need it once. After that, clear your terminal screen. 

Now we can run Git Commands using "git add ."; Commit - git commit -m "first commit"; 

**Push to GitLab** - git push -u origin main

(The actual branch name may vary depending on your setup.) This push depends on your internet connection, so it may take a little time.

Once the push is successful, go back to your GitLab repository in the browser and refresh the page. You will see that everything has been pushed successfully: the app directory, .gitignore, Dockerfile, app.py, and the Kubernetes file—all your files are now present in GitLab.

That’s how you perform code versioning with GitLab.

**6) GCP Setup - Service Accounts, GKE, GAR**

#### "Full Documentation.md" file of tutor in GitHub has every steps that we have to do for Deployment

In this video, we are setting up Google Cloud for our project. First, open Google Cloud Platform (GCP) and create an account. When you sign up, Google provides a $300 free credit. After creating your account, sign in and open your GCP console.

The first step is to enable six APIs. If you are using Google Cloud for the first time, these APIs are disabled by default. Enabling them ensures your project won’t face errors. To do this, go to the left-hand pane, select APIs and Services, and then select Library. Now search for and enable the following APIs one by one:

(i) Kubernetes Engine API – required for Kubernetes clusters.

(ii) Container Registry API – required for storing Docker images in Google Artifact Registry.

(iii) Compute Engine API – required for handling VM instance computation.

(iv) Cloud Build API – for building and deploying images.

(v) Cloud Storage API – for cloud-based storage.

(vi) IAM API – for managing roles and permissions.

Make sure each of these is enabled. In my case, they were already enabled, but you need to check yours carefully.

Next, we will create a Google Kubernetes Engine (GKE) cluster. From the GCP homepage, search for GKE and open Kubernetes Engine. In the left pane, select Clusters, then click Create Cluster. You will be asked to provide some details:

(i) Cluster name: Let’s name it lmops.

(ii) Region: Keep the default as us-central1.

(iii) Tier: Select the Standard tier (do not choose Enterprise, as it’s for company-level production projects).

(iv) Fleet registration: Skip this.

(v) Networking settings: Enable access using DNS and IPv4 address, but do not tick “Enable authorized networks.” Keep it unticked.

(vi) Advanced section: No changes required.

Now click Create. The cluster creation process usually takes around five minutes. You must wait until the cluster is fully created before proceeding.

While the cluster is being created, we can set up a Google Artifact Registry. Search for “Artifact Registry” in the GCP console and open it in a new tab. Create a new repository and name it lmops-repo. Select Docker as the format, set it to Standard, choose Region, and select the same region as your cluster (us-central1). Create the repository, which will be created instantly, unlike the cluster that takes longer.

**Next, we need to create a Service Account. This service account will allow our CI/CD tool (such as CircleCI, GitHub Actions, GitLab, or Jenkins) to partially access Kubernetes and Artifact Registry. Since we cannot directly use our personal account for deployment, the service account provides controlled access.**

From the left pane, go to IAM & Admin, then select Service Accounts. Create a new service account and give it any name (e.g., celebrity or any name of your choice). Click Create and Continue. Now assign the following five roles:

(i) Owner

(ii) Storage Object Admin

(iii) Storage Object Viewer

(iv) Artifact Registry Administrator

(v) Artifact Registry Writer

These five roles provide partial access to storage, registry, and Kubernetes operations. Once roles are added, click Done.

After creating the service account, go to its Actions menu and select Manage Keys. Add a new key, choose JSON format, and download it. This JSON file contains your access credentials.

Now open your VS Code, copy the downloaded JSON file into your project’s root directory, and rename it to gcp-key.json. Since this key must never be pushed to your repository, add gcp-key.json to your .gitignore file. This ensures the key remains private while still allowing your project to use it locally.

This key grants your project controlled access to GCP with the roles we assigned: Owner, Storage Object Admin, Storage Object Viewer, Artifact Registry Admin, and Artifact Registry Writer. These permissions allow Kubernetes and Artifact Registry to function properly during deployments.

To summarize, here’s what we completed in this video:

(i) Enabled six essential Google Cloud APIs.

(ii) Created a Google Kubernetes Engine cluster (wait until it’s fully created before continuing).

(iii) Created a Google Artifact Registry repository (lmops-repo).

(iv) Created a Service Account with five specific roles.

(v) Downloaded the JSON key, renamed it to gcp-key.json, and added it to .gitignore.

Finally, once your Kubernetes cluster is fully created, you will see a green tick mark, meaning the cluster is running successfully. Only then should you proceed to the next video.

**7. GitLab CI/CD Code:**

In this video, I will explain the code for setting up GitLab CI/CD.

First, in the root directory of your project, create a file named .gitlab-ci.yml. Once you create it, you will see a small box icon appear next to the filename, which indicates that the file has been named correctly. Press Enter, and now you have your GitLab CI/CD configuration file.

We are not writing this code from scratch. Instead, you can copy the code from the resource section of my GitHub and paste it into this file. In this video, I will only explain how the code works.

The first thing you will notice is that the base image is set to google/cloud-sdk:latest. If you remember from the CircleCI project, you already know why this is important. Although you could have used a Python image, the disadvantage is that Python images do not come with gcloud, kubectl, or docker pre-installed. Since these commands are used repeatedly in our pipeline, installing them separately would increase both setup and execution time. The Google Cloud SDK image already contains these tools, so it saves time in both coding and processing.

Our CI/CD pipeline has three stages:

(i) Checkout – for fetching the code.

(ii) Build – for building and pushing the Docker image.

(iii) Deploy – for deploying the Docker image to Google Kubernetes Engine (GKE).

In the Checkout stage, we don’t actually need to fetch anything because our code already resides in GitLab. Normally, if the source code was in GitHub or another SCM, we would copy it during this stage. In our case, we simply display a message confirming that the code is checked out.

The second stage is Build. Here, we build a Docker image of our application and push it to Google Artifact Registry. To do this, we use a feature called Docker-in-Docker (DinD). This is necessary because our base image itself runs in a Docker container (Google Cloud SDK). Inside that container, we also want to build our own Docker image for the application. By enabling docker:dind services, we can run Docker commands inside a Docker container.

Next comes the before_script section. Here, we configure Google Cloud authentication. In GitLab, we set an environment variable named GCP_SA_KEY that contains the content of the service account JSON file we created earlier. The pipeline converts this environment variable into a file named gcp-key.json. Using this key, we activate the Google Cloud service account and authenticate with our project. Then, we configure Docker authentication with our Artifact Registry.

After authentication, we build the Docker image using the defined registry path, project ID, repository name, and application name. For example, the image name format is:

<registry>/<project-id>/<repo>/<image-name>:latest

We then push this Docker image to Artifact Registry. It’s important that the image name here matches exactly with the image name defined in your Kubernetes deployment YAML file. For instance, if the deployment file specifies lmops:latest, then the same name must be used when building and pushing the image.

The third stage is Deploy. This stage is responsible for deploying the Docker image from Artifact Registry into the Kubernetes cluster. Similar to the build stage, we first authenticate with Google Cloud using gcp-key.json. Then we configure access to the Kubernetes cluster by fetching its credentials. For this step, we use the cluster name, cluster region, and project ID—all of which we defined as environment variables earlier.

Once cluster access is configured, we apply the kubernetes-deployment.yaml file. This file tells Kubernetes to pull the Docker image from Artifact Registry and deploy it inside the cluster. Again, make sure the image name inside the deployment file matches the name you used while pushing the image in the build stage.

**Summary of this:**

(i) We used Google Cloud SDK as the base image because it comes with gcloud, kubectl, and docker pre-installed.

(ii) The pipeline has three stages: Checkout, Build, and Deploy.

a) Checkout does nothing since the code already lives in GitLab.

b) Build stage creates a Docker image of the app and pushes it to Artifact Registry using Docker-in-Docker.

c) Deploy stage configures Kubernetes access and applies the kubernetes-deployment.yaml file to deploy the app.

(iii) Authentication is handled using a service account key (gcp-key.json) that we converted from the GitLab environment variable GCP_SA_KEY.

(iv) The image naming convention must remain consistent between the GitLab pipeline and the Kubernetes deployment file.

This completes the explanation of the GitLab CI/CD pipeline code.

**8. Full CI/CD Deployment on GKE**

First of all, you have to check some things. Your Docker file is there — yes, we have created the Docker file. Your Kubernetes deployment file is there — yes, we have created it. Have you done the code building using GitLab? Yes, we have done that, and I have also given you some instructions on how you can do the code versioning. But I think the video was enough for that purpose. Still, suppose someone wants to do the shortcut way, so that’s why I have given this as well.

In the previous videos we have done the GCP setup also. Basically, we enabled some API keys, created our Google Kubernetes cluster, created one Artifact Registry, and also created one service account along with the access key for that. Now here is our GCP key dot JSON, and make sure you include this GCP key dot JSON in your .gitignore. So, you have to write it like this: GCP key dot JSON. You must download it as a JSON file, place it properly, and make sure it’s excluded from Git tracking.

After that, you created this .gitlab-ci.yml. Now what you have to do is push this GitLab CI YAML to your GitLab. So, first of all, you must ensure you include the GCP JSON in .gitignore. Then, just like before, run the commands: git add, git commit, and git push. The same things you did earlier during the code versioning process. Once pushed, if I show you my GitLab, now you will see one .gitlab-ci.yml file is also there.

As soon as the .gitlab-ci.yml is there, you will get an option in GitLab settings. You can see under Settings, there is an option for CI/CD. We will be coming into CI/CD soon, but let’s go one by one. First of all, you have to convert this GCP key into Base64 format. Why Base64 format? Because in the GitLab CI YAML we are encoding the GCP key into Base64 format so that we can store it as an environment variable on GitLab. Later, GitLab will fetch this environment variable (GCP_SA_KEY), decode it back, and recreate the GCP key dot JSON to activate the service account. So first we encode, then we decode.

To do this, open a Git Bash terminal in VSCode. Copy the provided command, and make sure you open specifically a Git Bash terminal (not CMD or PowerShell). If your file is named GCP.json, use that exact name. If you saved it with some other name, adjust the command accordingly. Then press enter, and it will generate the Base64 string. Make sure you don’t include any extra spaces. Copy everything end-to-end without adding a space anywhere.

Now what you have to do after copying it is add this as a GitLab CI/CD variable. Go to Settings > CI/CD, then in the Variables section, add a new variable. Keep the type as Variable, environment as All, visibility default, and paste the Base64 value you copied earlier into the Value field. Again, check top and bottom for extra spaces — there should be none.

For the key, you must use the same name as written in the YAML. In this case, the key should be GCP_SA_KEY. Copy that from your VSCode and paste it into the Key field. Add the variable. You will see that the variable GCP_SA_KEY has been successfully added. Done — so you have added your CI/CD variable here.

Next, you have to set up your LM ops secrets in your Google Kubernetes cluster. Open your Google Kubernetes cluster, connect it, and run the Cloud Shell. You will get a command automatically to connect — just copy it and run it. This command will be like:

gcloud container clusters get-credentials <CLUSTER_NAME> --region <REGION> --project <PROJECT_ID>

By default, Google generates this for you in the Cloud Shell, so you can simply press Enter. If it doesn’t appear, I have also provided the command manually, and you can adjust <CLUSTER_NAME>, <REGION>, and <PROJECT_ID> according to your project.

Now, open your Kubernetes deployment YAML file. You will see environment variables defined inside, such as LM_OPS_SECRETS. Inside that, you have a key named API_KEY. To inject this secret into your cluster, open a Notepad and prepare the command:

kubectl create secret generic lmops-secrets --from-literal=API_KEY=<YOUR_API_KEY>

Here, replace <YOUR_API_KEY> with your actual key from .env. Since .env was never pushed to source control, we directly inject it into Kubernetes. Copy your key from .env in VSCode, paste it into the command, and then copy the entire command into your Cloud Shell. Press Enter, and you will see the message: Secret lmops-secrets created successfully. Done.

Now you are ready to trigger your CI/CD pipeline. First, make sure everything has been pushed to GitLab. Once checked, go to GitLab’s left pane → Build → Pipelines. From here, you can trigger the pipeline manually. Pipelines also run automatically whenever you push code, but you can start one manually if you want.

If you go into Settings > CI/CD, you will see pipeline triggers. By default, the trigger is a push event. But since we want to run manually now, go to Build → Pipelines, and create a new pipeline. It may ask for variables, but in our case, we already provided environment variables in GitLab, so just click Run pipeline.

Now your pipeline starts. It will have three stages: Checkout, Build, and Deploy. The Checkout stage will run first, then Build, then Deploy. This whole process may take around 5–10 minutes depending on your internet connection.

Once complete, you will see three green ticks. You can also click each stage to view logs if errors occur. In our case, there are no errors, so no need. With that, everything has passed successfully.

Now, go back to your Google Kubernetes Engine, close the Cloud Shell, and go to Workloads. At first, you may see an error: Does not have minimum availability. Don’t panic — this is normal while pods are being scheduled. Wait 5–6 minutes, and it will fix itself. The error message itself mentions that provisioning nodes may take a few minutes.

If you refresh after some time, you will see pods moving from Pending to Running. Once green ticks appear, it means the app is deployed. Alongside, you will get an endpoint. That endpoint opens your Streamlit application.

Sometimes it may not open instantly. Again, be patient. Wait for 2–3 minutes until the containers fully initialize. Then click the endpoint again, and your Streamlit application will load. If you see a No healthy upstream error, regenerate or refresh — these are small first-time initialization errors. Eventually, it will succeed.

Now test your app. For example, copy a prompt from README.md — “Compose a deeply emotional tempo evoking heartbreak and longing in a sad style” — and try generating music. At first, it may fail once, but on retry it will generate successfully. These minor hiccups are normal.

Finally, let’s talk about cleanup. To avoid charges, delete your Kubernetes cluster and Artifact Registry. Go to Clusters, select your cluster, delete it. Do the same with the Artifact Repository. The service account can stay — it won’t cost anything. GitLab project can also remain as it is. The only resources that could incur charges were the Artifact Registry and Kubernetes Engine, so deleting those two is enough.

And that’s it — your full CI/CD deployment pipeline is now complete.









**G) Multi AI Agent using,Jenkins,SonarQube,FastAPI,Langchain,Langgraph,AWS ECS**

**1. Introduction to the Project**

**Tech Stack**

(i) Groq - LLM

(ii) Tavily - Online Search Tool

(iii) LangChain - Generative AI Framework to interact with LLM

(iv) LangGraph - Agentic AI Framework to make AI Agents

(v) FastAPI - To build backend API's for handling user requests

(vi) Streamlit - To make UI or frontend of the app

(vii) Docker - For Containerization of the app during deployment

(viii) SonarQube - To check our code for Bugs, security issues, and bad practices. It's like a evaluator for the app

(ix) Jenkins - For making CI/CD Pipelines

(x) AWS ECS Fargate - To deploy and run app on Cloud without managing servers. It's a service offered by AWS

(xi) GitHub - SCM for the Project

This project is about building a multi-AI agent system where users can choose the type of agent they want, such as financial, medical, or study-related, and then ask domain-specific questions. The model will generate responses using Grok LM, while Tivoli APIs will be integrated to fetch real-time internet data, bridging the knowledge gap since most LMs are trained only up to 2021. For example, when asking about current stock trends, Tivoli provides up-to-date information that Grok can process.

The tech stack includes Grok LM (using the Llama 3 model), Tivoli for online search, LangChain for generative AI interaction, and LangGraph (built on LangChain) for agent workflows. FastAPI will be used for the backend to process requests, and Streamlit will handle the frontend for user interaction. The system flow is straightforward: user inputs are captured via the frontend, processed by the backend, and outputs are displayed back to the user.

To ensure scalability and deployment, Docker will containerize the application, enabling deployment on AWS services. Jenkins will be used to create a CI/CD pipeline that automates code fetching from GitHub, Docker image creation, image storage on AWS ECR (Elastic Container Registry), and deployment to AWS ECS Fargate. A load balancer (ALB) can be integrated to provide a static application URL instead of the default dynamic Fargate URL.

SonarQube plays a critical role as the project’s MVP for maintaining code quality. It identifies bugs, code smells (bad coding practices), and code duplication. Examples include detecting “dead code” that will never execute, flagging excessively long functions that hurt readability, and identifying repeated logic across files. This ensures the codebase is clean, maintainable, and production-ready.

The workflow starts with project setup, defining the structure, creating environment and requirements files, and configuring API keys for Grok and Tivoli. Then, logic is developed using LangChain and LangGraph, followed by backend and frontend integration. The project versioning is handled via GitHub, with Docker used for containerization. Jenkins automates the pipeline with stages like GitHub integration, SonarQube scanning, Docker image building, ECR pushing, and Fargate deployment. Finally, an Application Load Balancer ensures a static URL for consistent access.

In summary, this project combines Generative AI, Agentic AI frameworks, CI/CD pipelines, containerization, and cloud deployment into one cohesive application. The integration of SonarQube ensures high code quality, while Jenkins and AWS Fargate streamline deployment. The end result is a scalable, flexible, and user-friendly multi-agent system capable of providing domain-specific intelligent responses with up-to-date data.

**2. Project and API Setup ( Groq & Tavily )**

Hello everyone.

So this is your project setup video. It is a very important video for setting up your project structure, your API structure, your URL structure. So please watch the video carefully.

Okay.

So just open your any folder in your local PC and just create one folder. Here. Let me create a new folder. Let's name it multi AI agent. That is the theme of our project right. So just open it double click and just open in terminal okay. Right click and open in terminal. Here you will write code full stop.

So basically what this code will do. It will trigger your VS code okay. Just press enter and you will see VS code has been opened. Right now let's create our virtual environment for creating the virtual environment. What you will do basically just go here on the top bar you will see a terminal. Just click on New Terminal and from here select Command Prompt okay. Make sure it is command prompt right.

And in the command prompt section you will write:

python -m venv env


Basically it will create a virtual environment. Just press enter. On the right left side you will see env gets created okay. Now it will take some time. Uh 10 to 15 seconds I think. Then we will have to activate our virtual environment okay.

So let's activate our virtual environment. Let's wait. Okay, now we have to activate our virtual environment. How? You will activate. Just write whatever the name you have given to your environment. That is V env, then a backslash then scripts then backslash.

venv\Scripts\activate


Okay. Enter. Now you will see a Venv in the bracket. So it means your virtual environment has been created successfully. Okay, good.

Now, uh, we have to create one files here. Basically we will be defining our project structure now okay so let's create one file that is your requirements.txt. Now what this file do basically uh here you will list all the libraries that you need uh in your app. Okay. Let me close this terminal okay. Now what you will write inside this requirements.txt uh, basically all the libraries.

Right. We need first of all, uh, we need lang chain. Brock. Basically we will be using Brock API. So we need the Lang chain. Brock library. Okay. Then we need the Lang chain community library. So I will write length chain and I will write community. Okay then we need python dot env. So it also has a separate use case. I will tell you later. Okay then we have uvicorn. Basically it is for the API setup when we will create the backend of our app, we need this Uvicorn. Okay. Then we need the fast API for creating the API structure of our app. Okay. Then we need a library known as Pydantic also. Then we need Streamlit for creating the UI of our app, right? Then we need the Lang graph for creating AI agents. Okay, our project is multi agent. So how you will create an AI agent lang graph. Okay so this is all your all the requirements. You will find. You will find this file in the Uh, code section. I have uploaded the GitHub uh, link also, so you can just copy paste. Okay. No need to write, but I am teaching, so I have to write it from scratch. Okay. Good.

Now you will create one file here. Basically this file is your setup.py. So I have created a separate video for the setup.py code. So I will just copy paste the code for now. So if you want the explanation of how the setup.py file work, just go to that video and watch it. Okay. So here is your setup.py code. You will find on the code section. You will find on the GitHub section. Just copy from there and paste it in the setup.py section. Okay. Just change the file name of the project okay. Uh you can just write multi AI agent project whatever you want to do okay. Good.

Now just open the terminal. And just go to your previous terminal only and just write:

pip install -e .


Now basically it will do it will install all the requirements. Basically this code will trigger your setup.py and whatever the code inside this setup.py will trigger the requirements.txt installation. Okay. So you have to watch that setup.py code explanation. Then only you will understand okay. So just enter it and it will start installing all the libraries and project dependencies. So you can see it has started installing. Okay. So let it be.

Now we will create our project structure okay. Let's come to our project structure okay. So basically what you will do uh. We will create one app folder here okay. You will create one app folder here. Now we are all have project structure will be inside this folder. Okay, inside this app folder, your back end, your UI, your everything will be inside this app. Okay. So first of all inside app let's create folders. First of all our one folder will be backend. Inside this backend folder you will create your backend like your API setup everything okay. You will create one more folder here. That is your front end. So basically inside this there will be your UI and everything. Okay. Your Streamlit code, everything okay will be inside this.

Now you will create one folder one more folder here. First one is common okay. Now what is this common folder do uh basically here we will list our logging files and custom exception files. Okay. So you have to watch the separate video for custom exception file and logging files. Also if you want to understand, I have given in the uh before this project, I have given those files Explanation. Okay, so just watch it for that.

Now we will create one more folder here inside this app that is your folder. So basically in the core folder will be the main logic of our multi agent okay. Then we will create one more folder here. That is your config okay. So basically here will be the all the setup of your API. How to load the API and everything okay inside this config okay good. Everything good. So these are your all the folders okay. This is your project structure. Good.

Now inside this app you will create one file. Till now you have created only folders. Now you will create one file. And that will be your main.py. Okay. Inside this main.py will be your main app code. As soon as you create this main.py, your app will start running. Okay. So this is your main file. Basically you will connect your front end, back end and everything inside this main.py. Okay. And to make this app as a project dependency you will create one init file inside it. So you will create hyphen hyphen init hyphen hyphen dot pi. I have given the importance of this init inside the dot pi file setup.py code explanation. So you can watch that okay. So basically this file is needed so that you can import everything from one project directory to another project directory okay. So just create it and you can see your dependencies are also getting installed okay. So I think this is your only project structure.

As soon as we move ahead we will write the code for back end. Then we will write the code for front end, then for core config common and everything okay. Don't worry. Uh, yeah. For common we will write here only. So inside the common directory you will create one init file because we want in common also be treated as a package. Okay, so.py. Now inside common there will be two files. Okay. What are those two files. First of all your login file and your custom file. So first of all inside common let's create our logger.py. Then inside the common let's create our custom exception.py. Okay, now I have already explained the code for logger.py and custom exception.py in previous modules, so I will just copy paste the code. Okay, so just go to the GitHub directory that I have mentioned and just search for the yeah this logger.py and just copy it okay. Just copy it paste it in the VS code. Good okay. This will be used for logging. And just go here again for custom exception. This will be used for custom exception. Simple. Just copy paste. Okay. Done. Easy. Okay, good.

So that's how you have done your project setup. Okay. For this video, as soon as we move forward, like if you are doing backend we will do setup for backend. We will do setup for front end and everything. Okay so this is your basic project setup. Okay. And one more thing. You have to run that command again that you have to install hyphen E full stop. As soon as your all these dependencies get installed you have to run it again. Why? Because we have created one this one init file. Then inside common we have created one init file here. Every time you create one init file you have to run that command pip install hyphen e full stop.

Basically that pip install -e . will recognize your, uh, project directory as a package. Okay. So every time you create that init file you have to run that command pip install -e . okay. Only then your project will work properly. Good. Okay. You can see all the dependencies get installed. So just write that command again:

pip install -e .


Okay. So it is doing the installation. So basically what will happen I will tell you in short. So suppose if you want to import this logger uh okay. So this is your logger get logger function. So if you want to import this get logger function inside this front end directory. You will not be able to do. If you have not run this pip install -e . command okay. So if you have run this command then you will be able to do the imports from uh common directory to front end directory, common directory to backend directory. Okay. So that's the main use okay. Let me close the terminal. I think we don't have any use. Uh, so this is your project setup. Simple. Easy okay.

Now we will create, we will do our API setup. So how you will do your API setup? Basically in the project directory you will create one file .env okay. So this is your environment variable. Basically here you list your all the uh you can say API keys and everything okay that's the main use of env folder. So I will write we need two APIs. First of all we need grok grok API key okay grok API key. Then we need. API key okay. So these two will be used for our project grok API and API.

So I have explained this grok API and API in the project introduction video. So go watch that. So now just open your browser. Just search for grok okay. And just open it. Okay? And just make your account here. And just make your. Uh. Where this API. Oh, wait wait wait wait wait. I think I have to search graph API. Yeah. Okay, so this is your API keys. Just make your account here. So I have already logged in into my account. It is free. Okay. So just go to your API keys and create your API key. Okay. Let me delete my previous. And let's create one API key. And you can give any name. Uh let's keep it LM ops okay. And submit. And it has given you the API key. Make sure you copy it safely and just copy it and go to our VS code and paste it inside this grok API key. Done. Easy. Okay, so now you have done your grok API setup.

Now let's search for another one that is your API. Okay. Just go to the telecom that is your first site login okay just login. I have already logged in so it automatically get me logged in. And in the overview section you will get your API key. That's the by default API key. Just copy it from here okay. And just go in the VS code and just paste it here okay. Easy. So these are the only two API's that you need for this project grok API. Basically it will be used for uh your LMS and API. It will basically use for your search online search okay. Okay. So these are your two API's.

So your project structure is done. Your API setup is done. Now the third one. Your setup. Okay. So now let's do our setup. So basically how you will do your setup. What is your your Windows Subsystem Linux. Uh why we are using SSL. Because when you do your production grade projects. Right. So you will be doing that projects in a Linux environment, not in a windows environment. So we will give you a feeling of Linux inside a windows system okay. That's why we need uh WSL okay.

So how you will do basically in my PC WSL is already set up. So it is hard to explain you how to install it. But I will give you a very easy method. So basically why we want we want WSL with Docker Engine okay. So that you can create your Docker apps and you can create your Jenkins pipeline in the ahead of the project. So that's why we need the WSL. So just go to the, uh, whatever. I have mentioned the GitHub documentation there. Okay. Just go there. And there you will find one this full documentation.md file. Okay. So in the starting only you will get an installation tutorial.

So first of all open your PowerShell okay. Every windows have PowerShell. So open it in administrator mode. Administrator mode okay. And you will write this command will install. Just copy it and run it on your PowerShell administrator. So it will basically install your WSL if not installed if already installed, if you think you have already installed. So then just write this command WSL update and reboot your system. Restart your system, basically restart your PC. Uh, although it's not necessary, but for your safety, restart your PC. Okay, now we will be using a ubuntu environment. So what we will do basically just go to the Microsoft Store okay. Every window have a Microsoft store, so just go here and search for ubuntu. Okay. Here search for ubuntu. Okay. Just search. Okay. And now just search for ubuntu 22.40. Just open it. So first of your first one only. So this one is ubuntu. Just install it. Okay I have already installed. That's why it is showing me open but already installed okay. Okay. You can install anyone if you want this ubuntu 22.24 .404. But I will tell you, this one is the best. That one your normal ubuntu. Okay, this one is best. Just install it. Okay. Open it and check and install everything. Okay.

Once it is installed, click it and just proceed with the steps. It will guide you through the steps. Install okay. Simple. Now, uh, you have to, uh, you're going to set up. Your cell is set up. Now, what you have to do, basically, you have to install Docker inside your ubuntu, Docker inside ubuntu. So you can just search on Docker installation on ubuntu okay. Install Docker engine on ubuntu. So basically it works as same as installing Docker desktop. You know there is a app for Docker Docker desktop right? I have used in the uh other projects also Docker desktop. But for this project we are using Docker Engine on Ubuntu using WSL. Okay. So just follow the requirements okay. So it wants it wants uh ubuntu 24.04 that will be installed. Okay. Now uh, just run this command one by one inside ubuntu. Just open ubuntu like this. Like I have already installed. So once your ubuntu is installed it will be like this. Just open it. Okay, so open it and it is taking some time. Okay. Then. Uh, now run those command line by line. Okay. First of all this setup dot docker app repository, then these these these. Basically I have given these command here also okay. So you can if you want to do it from here you can just do it okay.

Now once you have done restart the ubuntu terminal just cancel this ubuntu terminal and restart it again okay. And just uh if you want to make sure it is installed or not, just write ubuntu in the ubuntu terminal, just write Docker version version, press enter and you will see here Client Docker Engine community 28.1 Docker Engine installed server. Docker engine also installed. Okay, so basically Docker Engine comprises of two things client and server. Okay. So these two things should be installed if you want to basically check. So you will write:

docker --version


and you will see Docker version this this this this this. So if it is showing you something like this it means your docker have been successfully installed. If not, so you have uh, done some mistake in some of the steps. Okay. Uh, it's not a big deal. Okay. You can easily counter this. Okay, so with this, your Docker setup is also done. Okay. So this was it for this video.

So basically what you have done you have setup your project structure virtual environment logging custom exceptions files your API setup your setup. All done. So let's move to the next video.

**Summary:**

(i) Create Project Folder & Open in VS Code

Create a folder multi AI agent.

Open it in terminal and launch VS Code using code ..

(ii) Create Virtual Environment

Run: python -m venv env

Activate it: venv\Scripts\activate.

Confirm activation (shows (venv) in terminal).

(iii) Create requirements.txt

Add required libraries:

langchain-brock

langchain-community

python-dotenv

uvicorn

fastapi

pydantic

streamlit

langgraph

(iv) Create setup.py

Copy-paste setup code (provided separately in GitHub).

Update project name if needed.

Install dependencies with:

pip install -e .


(v) Define Project Structure (app/ folder)

Inside app/ create folders:

backend/ → for API code.

frontend/ → for UI (Streamlit).

common/ → for logging & custom exceptions.

core/ → main multi-agent logic.

config/ → API setup & configurations.

Inside app/, add:

main.py → connects everything.

__init__.py → make package imports work.

(vi) Setup common/ folder

Add __init__.py.

Create logger.py (logging logic).

Create custom_exception.py (custom error handling).

Run again:

pip install -e .


(needed whenever you add new __init__.py).

(vii) Environment Variables

Create .env file in root.

Add:

GROK_API_KEY = <your_grok_api_key>
API_KEY = <your_api_key>


Get keys:

Grok API (for LLM).

Another API (for online search).

(viii) WSL & Docker Setup

Install WSL (Windows Subsystem for Linux) using PowerShell (Admin):

wsl --install
wsl --update


Restart system.

Install Ubuntu 22.04 from Microsoft Store.

Setup Docker Engine inside Ubuntu:

Follow official Docker docs (add repo, install Docker packages).

Verify with:

docker --version


(both Client & Server should be installed).

**What we did:**

(i) Project folder & environment ready.

(ii) Dependencies installed.

(iii) Project structure (app/ with subfolders) created.

(iv) Logging & exception handling setup.

(v) API keys configured in .env.

(vi) WSL + Docker setup complete for production-like environment.

**3. Configuration Code**

In this lecture, we will be setting up our configuration file. First, open your app directory and go to the config directory. To make this directory behave like a Python package, create a file named __init__.py. After that, create another file inside config named settings.py, which will hold all of our configuration code.

Inside settings.py, we first need to import the required libraries. For this, write "from dotenv import load_dotenv" and "import os". The dotenv library is useful because it allows us to load the values from our .env file, such as API keys. The OS library is required so we can interact with our operating system and read those values.

Next, we load the environment variables by calling "load_dotenv()". This ensures that all the keys stored inside your .env file are accessible within your settings file.

Now, let’s create a class called Settings. Inside this class, we will define variables to store our keys. For example, write "class Settings:" and then "grok_api_key = os.getenv('GROK_API_KEY')" followed by "table_api_key = os.getenv('TABLE_API_KEY')". Remember, the variable names inside os.getenv() must exactly match the names inside your .env file—any spelling mistake will cause errors.

After loading the keys, we also want to specify which models are allowed for use. For this, inside the class, create a list by writing "allowed_models = ['llama-3-70b', 'llama-3-8b-instant']". These model names must be copied exactly as they are provided by the API, otherwise they won’t work. The larger 70B model is more accurate, while the 8B instant version is faster but less accurate.

Finally, outside the class, we create an object to load it by writing "settings = Settings()". This way, whenever you want to access your API keys or allowed models in your project, you can simply import them from "config.settings".

This completes our settings.py configuration file. In the next step, we will see how to use these settings throughout the project.

**Summary:**

Configuration File Setup (config/settings.py)

(i) Make config/ a Python Package

Inside config/ folder, create __init__.py (empty file).

This allows importing from config like a Python package.

(ii) Create settings.py

This file will hold all configuration variables (API keys, models, etc.).

(iii) Import Required Libraries

from dotenv import load_dotenv
import os
dotenv → to read environment variables from .env file.

os → to interact with the system and get environment variable values.

(iv) Load Environment Variables

python
Copy code
load_dotenv()
Ensures all keys in .env file are accessible in Python.

(v) Create Settings Class

python
Copy code
class Settings:
    grok_api_key = os.getenv('GROK_API_KEY')
    table_api_key = os.getenv('TABLE_API_KEY')
    allowed_models = ['llama-3-70b', 'llama-3-8b-instant']
grok_api_key → stores Grok API key.

table_api_key → stores Table API key.

allowed_models → list of models allowed in the project:

llama-3-70b → larger, more accurate.

llama-3-8b-instant → faster, less accurate.

(vi) Instantiate Settings

settings = Settings()


Creates a settings object that can be imported anywhere in the project.

Example usage elsewhere:

from config.settings import settings
print(settings.grok_api_key)

**What we did:**

(i) Centralized configuration file for all keys and settings.

(ii) Easy access throughout the project via settings object.

(iii) Ensures API keys and allowed models are properly loaded and managed.

**4. Core Code:**

In this video, we will be setting up the core code for our app. First, open your app directory and go to the core directory. To make this directory behave like a package, create a file named "__init__.py". Remember, every time you create an "__init__.py" file, you should run "pip install -e ." in your terminal to initialize all dependencies properly.

Next, inside the core directory, create a file named "ai_agent.py". This file will contain all the core logic for your app. Before writing the code, ensure your dependencies are installed by running "pip install -e ." and waiting for it to complete.

Now, let’s start writing the code. First, import the required modules. For Groq models, write "from langchain.brock import Chad". The Chad class is necessary to use the Brock models, such as the LLaMA 70B model or the versatile LLaMA models. For Table API, write "from langchain_community.tools.tivoli_search import TivoliSearchResults". These imports ensure both our APIs’ tools are ready for use.

Since our project is multi-agent, we will also use LangGraph to create AI agents. Import it by writing "from langgraph.prebuilt import create_react_agent". Additionally, import "from langchain.core.messages.ai import AIMessage" to differentiate between AI-generated messages and human messages in the conversation. If LangChain Core is not installed, add it to your requirements.txt and run "pip install -e ." again.

We also need to import the configuration file. Write "from app.config.settings import settings". This gives us access to our Groq API key, Table API key, and allowed model names.

Now, let’s create a function called "get_response_from_ai_agents". This function will take parameters such as "lm_id" (the model ID, e.g., LLaMA 70B or LLaMA versatile), "query" (the user’s question), "allow_search" (whether to enable online search using Table API), and "system_prompt" (instructions to control AI behavior).

Inside this function, first set up the language model by writing "lm = Chad(model=lm_id)". This ensures the model corresponds to the user-selected LM. For tools, write "tools = [TivoliSearchResults(max_results=2)] if allow_search else []". This checks if online search is allowed and limits results to the top two items if enabled.

Next, create the agent using "agent = create_react_agent(model=lm, tools=tools, state_model_file=system_prompt)". Here, the state_model_file parameter sets the AI’s system prompt to guide its behavior.

Start the conversation history by writing "state = {'messages': query}". This keeps track of all messages the user has sent so far. Then, run the agent with "response = agent.invoke(state)". The agent will read all messages, process the query, optionally use web search, and generate a reply.

To extract only the AI-generated responses, write "messages = response.get_any_messages()" and filter using a list comprehension: "ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]". This ensures we ignore human messages and keep only the AI’s replies.

Finally, return only the latest reply by writing "return ai_messages[-1]". Using -1 indexing fetches the most recent message from the list.

This completes our AI agent core code. We will use the "get_response_from_ai_agents" function in both the back-end and front-end of our app in upcoming videos.

**Summary:**

Core Code Setup (core/ai_agent.py)

(i) Make core/ a Python Package

Create __init__.py inside core/.

Every time you add a new __init__.py, run:

pip install -e .


to reinitialize dependencies.

(ii) Create ai_agent.py

This file contains the main logic for handling AI agents.

(iii) Import Required Modules

from langchain.brock import Chad
from langchain_community.tools.tivoli_search import TivoliSearchResults
from langgraph.prebuilt import create_react_agent
from langchain.core.messages.ai import AIMessage
from app.config.settings import settings


Chad → to access Brock/LLaMA models (70B or versatile).

TivoliSearchResults → to use Table API for online search results.

create_react_agent → to build multi-agent behavior with LangGraph.

AIMessage → to separate AI messages from human messages.

settings → to access API keys and allowed model names from config.

(iv) Define get_response_from_ai_agents Function

def get_response_from_ai_agents(lm_id, query, allow_search=False, system_prompt=None):
    # Setup language model
    lm = Chad(model=lm_id)
    # Configure tools (Table API search)
    tools = [TivoliSearchResults(max_results=2)] if allow_search else []
    # Create AI agent
    agent = create_react_agent(model=lm, tools=tools, state_model_file=system_prompt)
    # Initialize conversation state
    state = {'messages': query}
    # Invoke agent
    response = agent.invoke(state)
    # Extract only AI-generated messages
    messages = response.get_any_messages()
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    # Return the latest AI reply
    return ai_messages[-1]

(v) Function Parameters

lm_id → ID of the language model (e.g., LLaMA 70B).

query → user’s input question.

allow_search → enable/disable online search with Table API.

system_prompt → instructions for AI behavior.

(vi) Function Workflow

Selects LM based on lm_id.

Initializes tools if online search is allowed.

Builds a LangGraph agent with model + tools + system prompt.

Maintains conversation history (state).

Runs the agent and gets the response.

Filters out only AI-generated messages.

Returns the latest AI reply.

**What we did:**

(i) Core AI logic is centralized in ai_agent.py.

(ii) Multi-agent support using LangGraph.

(iii) Optional web search via Table API.

(iv) Easy integration in back-end and front-end using get_response_from_ai_agents.

**5. Backend using Fast API**

In this video, we will be setting up the back-end code for our app. First, navigate to your app directory, and inside the back-end section, create an "__init__.py" file to make the directory a package. Remember to run "pip install -e ." again in the terminal so all dependencies are properly initialized.

Next, create a file named "api.py" inside the back-end folder. This file will contain the FastAPI application for our multi-AI agent. Begin by importing the necessary modules. Write "from fastapi import FastAPI, HTTPException" to use FastAPI and handle HTTP exceptions for errors such as 500, 404, or 200 status codes. For data validation, import "from pydantic import BaseModel" which ensures that incoming requests follow a valid structure. Also, import "from typing import List" to define list types in your request data.

We then import our core logic and configuration by writing "from app.core.ai_agent import get_response_from_ai_agents" and "from app.config.settings import settings". For logging, import "from app.common.logger import get_logger" and initialize it with "logger = get_logger(__name__)". Initialize the FastAPI app itself using "app = FastAPI(title='Multi AI Agent')" to give it a title.

Next, define the structure of incoming requests with a class "RequestState(BaseModel)". The class should include "model_name: str" to specify the language model, "system_prompt: str" for instructions guiding AI behavior, "messages: List[str]" to hold the conversation history, and "allow_search: bool" to indicate whether online search is enabled. This ensures all incoming data is validated properly and prevents errors downstream.

Now, create a POST endpoint for handling chat requests. Write "@app.post('/chat')" and define a function "def chat_endpoint(request: RequestState)". Inside this function, log the received request using "logger.info(f'Received request for model: {request.model_name}')". Check if the provided model name is in the list of allowed models with "if request.model_name not in settings.allowed_model_names". If the model is invalid, log a warning "logger.warning('Invalid model name')" and raise an exception with "raise HTTPException(status_code=400, detail='Invalid model name')".

If the model is valid, use a try-except block to call the core function and generate the AI response. Write "response = get_response_from_ai_agents(lm_id=request.model_name, query=request.messages, allow_search=request.allow_search, system_prompt=request.system_prompt)". Log the successful response with "logger.info(f'Successfully got response from AI agent: {request.model_name}')", and return it as JSON with "return {'response': response}".

In case of any unexpected errors, handle exceptions with "except Exception as e:" and log the error using "logger.error('Error occurred during response generation')". Raise a custom exception with "raise HTTPException(status_code=500, detail=str(CustomException('Failed to get AI response', error_detail=e)))". This ensures the API consistently returns structured error messages if anything goes wrong.

Overall, this back-end setup includes imports for FastAPI, Pydantic, typing, core AI logic, settings, logging, and custom exceptions, initialization of the logger and FastAPI app, validation of incoming requests, and a /chat endpoint that handles requests, logs activity, calls the AI agent, and returns responses while handling errors gracefully.

**Summary:**

Back-End Setup (backend/api.py)

(i) Make backend/ a Python Package

Create __init__.py inside backend/.

Run:

pip install -e .


whenever a new __init__.py is added to initialize dependencies.

(ii) Create api.py

This file handles the FastAPI application and endpoints for the multi-AI agent.

(iii) Import Required Modules

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.core.ai_agent import get_response_from_ai_agents
from app.config.settings import settings
from app.common.logger import get_logger


FastAPI & HTTPException → to create API routes and handle errors.

Pydantic BaseModel → validates incoming request structure.

List → defines list-type fields in requests.

Core & Config Imports → access AI agent logic and API keys/models.

Logger → to log activity and errors.

(iv) Initialize Logger & FastAPI App

logger = get_logger(__name__)
app = FastAPI(title='Multi AI Agent')


(v) Define Request Model

class RequestState(BaseModel):
    model_name: str          # Language model to use
    system_prompt: str       # Instructions for AI behavior
    messages: List[str]      # Conversation history
    allow_search: bool       # Enable/disable online search


Ensures proper validation of incoming requests.

(vi) Create POST /chat Endpoint

@app.post('/chat')
def chat_endpoint(request: RequestState):
    logger.info(f'Received request for model: {request.model_name}')
    # Validate model
    if request.model_name not in settings.allowed_models:
        logger.warning('Invalid model name')
        raise HTTPException(status_code=400, detail='Invalid model name')
    try:
        # Call core AI agent
        response = get_response_from_ai_agents(
            lm_id=request.model_name,
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt
        )
        logger.info(f'Successfully got response from AI agent: {request.model_name}')
        return {'response': response}
    except Exception as e:
        logger.error('Error occurred during response generation')
        raise HTTPException(status_code=500, detail=str(CustomException('Failed to get AI response', error_detail=e)))


(vii) Endpoint Workflow

Logs incoming request.

Checks if the model is allowed.

Calls get_response_from_ai_agents to generate a reply.

Returns AI response as JSON: {'response': <AI reply>}.

Handles errors with proper logging and structured HTTP exceptions.

**What we did:**

(i) Fully functional FastAPI back-end for multi-AI agents.

(ii) Request validation with Pydantic.

(iii) Logging and error handling implemented.

(iv) Integrates seamlessly with core/ai_agent.py and configuration (settings.py).

**6. Frontend using Streamlit**

In this video, we will be setting up the front-end code for our multi-AI agent app. First, navigate to your app/front_end directory and create an "__init__.py" file. After creating it, always run "pip install -e ." in the terminal to ensure that the package is properly installed.

Next, create a file called "ui.py", which will contain all the UI code. Start by importing the necessary libraries: "import streamlit as st" for building the user interface, and "import requests" to send HTTP POST requests to your back-end API. Additionally, import the settings and logger with "from app.config.settings import settings", "from app.common.logger import get_logger", and "from app.common.custom_exception import CustomException". Initialize the logger using "logger = get_logger(__name__)".

Configure the Streamlit page using "st.set_page_config(page_title='Multi AI Agent', layout='centered')" and give it a header title with "st.title('Multi AI Agent using Llama')" or any preferred title.

Next, define the system prompt, which specifies the behavior of your AI agent. Use "system_prompt = st.text_area('Define your AI agent', height=70)" to allow the user to specify whether the agent is a medical assistant, business helper, homework guide, or general knowledge agent.

Then, allow the user to select a model using "selected_model = st.selectbox('Select your AI model', settings.allowed_model_names)", where the allowed models are defined in your settings. Add a checkbox for web search functionality using "allow_web_search = st.checkbox('Allow web search')".

Collect the user query with "user_query = st.text_area('Enter your query', height=150)". The user query is separate from the system prompt; the system prompt defines the AI agent’s behavior, while the query contains the actual questions to ask the AI.

Define the API URL where your back-end is running, e.g., "api_url = 'http://127.0.0.1:9999/chat'", which points to the chat endpoint of your FastAPI back-end.

Create a button to trigger the query with "if st.button('Ask Agent'):". Inside this block, first strip the user query with "user_query.strip()" to remove extra spaces. Then create a payload dictionary to send to the API:

payload = {
    'model_name': selected_model,
    'system_prompt': system_prompt,
    'messages': [user_query],
    'allow_search': allow_web_search
}


Send the payload using "response = requests.post(api_url, json=payload)". If the API returns a success status code 200, extract the AI agent response using "agent_response = response.json().get('response', '')" and log it with "logger.info('Successfully received response from backend')". Display the response in the UI with "st.subheader('Agent Response')" and "st.markdown(agent_response.replace('\n', '<br>'), unsafe_allow_html=True)" to replace newlines with HTML break tags for cleaner formatting.

If the API request fails (e.g., 404 or 500), log the error with "logger.error('Backend error')" and show an error in Streamlit with "st.error('Backend error')". Wrap the request in a try-except block to handle any unexpected errors with "except Exception as e:", log the exception, and show a message using "st.error('Custom exception: Failed to communicate with backend')".

In summary, this front-end code handles:

(i) Importing Streamlit, requests, settings, logger, and custom exceptions.

(ii) Configuring the page layout and title.

(iii) Collecting system prompt, model selection, allow web search, and user query from the user.

(iv) Sending this data as a JSON payload to the back-end API.

(v) Receiving and displaying the AI agent response in a clean, formatted way.

(vi) Handling exceptions and API errors gracefully.

This sets up the UI for the multi-AI agent app. In the next video, you will see the front-end connected to the back-end and running live.

**Summary:**

(i) Front-End Setup (front_end/ui.py)

Make front_end/ a Python Package

Create __init__.py inside front_end/.

Run:

pip install -e .


whenever a new __init__.py is added to initialize dependencies.

(ii) Create ui.py

This file contains all the Streamlit UI logic for the multi-AI agent app.

(iii) Import Required Modules

import streamlit as st
import requests
from app.config.settings import settings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


Streamlit → builds the user interface.

Requests → sends POST requests to back-end API.

Settings → access allowed models and API configuration.

Logger & CustomException → logging and exception handling.

(iv) Initialize Logger & Configure Page

logger = get_logger(__name__)
st.set_page_config(page_title='Multi AI Agent', layout='centered')
st.title('Multi AI Agent using Llama')


(v) Collect System Prompt & User Input

system_prompt = st.text_area('Define your AI agent', height=70)
selected_model = st.selectbox('Select your AI model', settings.allowed_models)
allow_web_search = st.checkbox('Allow web search')
user_query = st.text_area('Enter your query', height=150)


System prompt → defines AI agent behavior (e.g., medical assistant, business helper).

Model selection → choose from allowed models in settings.

Web search option → enable/disable online search.

User query → actual question to ask the AI.

(vi) Define API Endpoint

api_url = 'http://127.0.0.1:9999/chat'


(vii) Trigger Query with Button

if st.button('Ask Agent'):
    try:
        user_query = user_query.strip()
        payload = {
            'model_name': selected_model,
            'system_prompt': system_prompt,
            'messages': [user_query],
            'allow_search': allow_web_search
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            agent_response = response.json().get('response', '')
            logger.info('Successfully received response from backend')
            st.subheader('Agent Response')
            st.markdown(agent_response.replace('\n', '<br>'), unsafe_allow_html=True)
        else:
            logger.error('Backend error')
            st.error('Backend error')
    except Exception as e:
        logger.error(f'Custom exception: {e}')
        st.error('Custom exception: Failed to communicate with backend')


(viii) Front-End Workflow

Collect system prompt, model selection, web search preference, and user query.

Create JSON payload and send POST request to /chat endpoint.

Display AI agent response in a clean format using Streamlit.

Log successes and handle exceptions gracefully.

**What we did:**

(i) Fully functional Streamlit front-end for multi-AI agent.

(ii) Dynamic model selection and system prompt input.

(iii) Connects seamlessly with FastAPI back-end.

(iv) Handles errors and exceptions gracefully for smooth user experience.

**7. Main Application Code**

In this video, we connect the front-end and back-end so that the app runs successfully. Start by opening the "main.py" file created during the project setup. This file will orchestrate running both the front-end and back-end together.

First, import the required modules: "import subprocess", "import threading", and "import time", which are all built-in Python libraries. Then import the logging and custom exception utilities using "from app.common.logger import get_logger" and "from app.common.custom_exception import CustomException". Initialize the logger with "logger = get_logger(__name__)".

Next, import "load_dotenv" from the dotenv package using "from dotenv import load_dotenv" and call "load_dotenv()" in "main.py". This ensures that all environment variables are loaded when running "main.py" in production, even if they were already loaded in "settings.py".

We then create two functions: one to run the back-end and one to run the front-end.

Back-end function:

def run_backend():
    try:
        logger.info("Starting backend service")
        subprocess.run([
            "uvicorn",
            "app.backend.api:app",
            "--host", "127.0.0.1",
            "--port", "9999"
        ], check=True)
    except Exception as e:
        logger.error("Problem with backend service")
        raise CustomException("Failed to start backend") from e


This function triggers the FastAPI back-end using Uvicorn on host "127.0.0.1" and port "9999". Errors are logged and raised as a custom exception.

Front-end function:

def run_frontend():
    try:
        logger.info("Starting front-end service")
        subprocess.run([
            "streamlit", "run", "app/front_end/ui.py"
        ], check=True)
    except Exception as e:
        logger.error("Problem with front-end service")
        raise CustomException("Failed to start front-end") from e


This function triggers the Streamlit front-end by running "app/front_end/ui.py". Exceptions are also handled similarly.

Next, we merge both functions so that the back-end runs asynchronously, allowing the front-end to start independently. This is achieved using threads:

if __name__ == "__main__":
    try:
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.start()
        time.sleep(2)  # Give backend time to start
        run_frontend()
    except CustomException as e:
        logger.exception(f"Custom exception occurred: {e}")
        print(str(e))


Here, a separate thread runs the back-end first, followed by a short sleep (2 milliseconds) to ensure the back-end is ready before the front-end starts. Then the front-end is triggered. Exceptions during startup are logged and displayed.

Once "main.py" is run using "python app/main.py", both the FastAPI back-end (on port 9999) and the Streamlit front-end (on port 8501) start automatically. You can now define your AI agent, select the model, allow web search, and enter user queries.

For example, if you set the agent to a medical AI agent specialized in cancer, it will only answer queries relevant to that domain. Queries outside the domain (e.g., general stock advice) will be filtered or constrained to the agent’s specialization. You can also experiment with different AI models like "Llama 70B Versatile" or "Mistral" depending on your allowed models in settings.

This setup ensures your multi-AI agent app runs seamlessly, with the back-end API handling queries and the front-end providing a user-friendly interface. The two ports engaged are 8501 (Streamlit) and 9999 (FastAPI).

**Summary:**

**Main Orchestrator Setup (main.py)**

(i) Purpose

Orchestrates running both the FastAPI back-end and Streamlit front-end together.

Ensures smooth communication and proper startup of both components.

(ii) Import Required Modules

import subprocess
import threading
import time
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from dotenv import load_dotenv


subprocess → runs back-end and front-end as separate processes.

threading → allows back-end to run asynchronously while front-end starts.

time → gives a delay to ensure back-end is ready before front-end starts.

Logger & CustomException → for logging and error handling.

load_dotenv → loads environment variables from .env file.

(iii) Initialize Logger and Load Environment Variables

logger = get_logger(__name__)
load_dotenv()


(iv) Define Back-End Function

def run_backend():
    try:
        logger.info("Starting backend service")
        subprocess.run([
            "uvicorn",
            "app.backend.api:app",
            "--host", "127.0.0.1",
            "--port", "9999"
        ], check=True)
    except Exception as e:
        logger.error("Problem with backend service")
        raise CustomException("Failed to start backend") from e


Runs FastAPI back-end using Uvicorn on 127.0.0.1:9999.

Logs errors and raises them as custom exceptions.

(v) Define Front-End Function

def run_frontend():
    try:
        logger.info("Starting front-end service")
        subprocess.run([
            "streamlit", "run", "app/front_end/ui.py"
        ], check=True)
    except Exception as e:
        logger.error("Problem with front-end service")
        raise CustomException("Failed to start front-end") from e


Runs Streamlit front-end using ui.py.

Handles exceptions and logs errors.

(vi) Run Back-End and Front-End Together

if __name__ == "__main__":
    try:
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.start()
        time.sleep(2)  # Give backend time to start
        run_frontend()
    except CustomException as e:
        logger.exception(f"Custom exception occurred: {e}")
        print(str(e))


Back-end runs in a separate thread.

Adds a short delay (time.sleep(2)) to ensure back-end is ready before front-end starts.

Front-end runs independently after back-end initialization.

Logs and prints exceptions during startup.

(vii) Outcome

Running python app/main.py starts both services automatically:

Back-end → FastAPI on port 9999.

Front-end → Streamlit on port 8501.

Users can now define their AI agent, select a model, enable web search, and enter queries.

AI agents respond based on system prompt and allowed models, with queries outside the specialization filtered.

**What we did:**

(i) Asynchronous back-end ensures front-end doesn’t wait unnecessarily.

(ii) Threading + subprocess handles process management.

(iii) Environment variables and logging make the app production-ready.

(iv) Seamless integration between UI and AI agent logic.

**8. Code Versioning**

In this video, we cover code versioning, which is the process of uploading and managing your code using a source version management (SVM) system. In our case, we are using GitHub. Code versioning allows you to track changes, build versions, and maintain a history of your project.

First, in the root directory of your project, create a file named ".gitignore". This file specifies all the files and folders you don’t want to push to GitHub. Typically, you include the following:

"venv/" – your virtual environment folder, because it is very large (1–1.5 GB) and exceeds GitHub limits.

Project management files – any project-specific files you don’t want to upload.

Logs – optional; you can include them if needed.

".env" – contains sensitive information like API keys, so it should never be pushed.

Save the .gitignore file after listing these entries.

Next, create a GitHub account and a new repository. In this example, we name it "multi AI agent project". Do not add a README file initially, as this may conflict with the initial push commands.

Download Git CLI from the official Git website according to your operating system (Windows in our case). Git CLI allows you to run Git commands in your terminal.

Once installed, open a terminal in VS Code and run the following Git commands step by step:

Initialize Git in your project folder:

git init


Rename your main branch to "main":

git branch -M main


Connect your local repository to the GitHub repository:

git remote add origin <repository-URL>


If a remote origin already exists, remove it first using:

git remote remove origin


Next, add all files for commit:

git add .


Commit the changes with a message:

git commit -m "Initial commit"


Push your code to GitHub:

git push origin main


Once done, refresh your GitHub repository page, and you will see all your project files uploaded, including directories like "app", "requirements.txt", setup files, and .gitignore.

This process ensures that your code is properly versioned and tracked in GitHub. You can follow the same pattern for future projects, adjusting the project name as needed, while always excluding sensitive files like venv/ and .env.

**Summary:**

(i) Purpose

Track changes, maintain project history, build versions, and manage code using GitHub.

Prevent accidental upload of sensitive files like API keys or large files.

(ii) Create .gitignore File

In the root directory of your project, create a file named .gitignore.

Add entries for files/folders you don’t want to push:

venv/             # Virtual environment (1–1.5 GB)
*.log             # Log files (optional)
.env              # Sensitive environment variables
Project-specific management files (optional)


Save the .gitignore file.

(iii) Set Up GitHub Repository

Create a GitHub account (if not already done).

Create a new repository, e.g., multi AI agent project.

Do not add a README initially to avoid push conflicts.

(iv) Install Git CLI

Download Git CLI from the official Git website for your OS (Windows in this case).

Git CLI enables running Git commands from your terminal.

(v) Initialize Git in Your Project

Open terminal in VS Code.

Run the following commands step by step:

git init                             # Initialize Git in project folder
git branch -M main                    # Rename main branch to "main"
git remote add origin <repository-URL>  # Connect local repo to GitHub


If a remote already exists, remove it first:

git remote remove origin


(vi) Commit and Push Your Code

Add all project files for commit:

git add .


Commit the changes with a message:

git commit -m "Initial commit"


Push the code to GitHub:

git push origin main


(vii) Verify Upload

Refresh your GitHub repository page.

You should see all project files, including:

app folder

requirements.txt

Setup files

.gitignore

(viii) Best Practices

Always exclude sensitive files (.env) and large folders (venv/).

Follow this pattern for future projects, updating the project name and repository URL accordingly.

**What we did:**

(i) Ensures version control, backup, and collaboration.

(ii) Keeps sensitive information and large files safe.

(iii) Provides a clear history of project changes for future reference.

**9. Dockerfile**

In this video, we are creating a Docker file for the project. Start by creating a file named Dockerfile in the root directory. To save time, the instructor copies the code directly and explains it line by line. You can also copy the Docker file from the provided GitHub link or code resources.

The Docker file starts with specifying a base image:

FROM python:3.11-slim


This is the parent image we will use for the project. We choose Python 3.11 slim because it is a stable, lightweight image suitable for deployment. Avoid using unstable Python versions like 3.13.

Next, some environment variables are set:

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


These are essential for production-grade projects. PYTHONUNBUFFERED=1 ensures that logging is handled in real time, which helps in debugging and monitoring.

The Docker file then defines a working directory:

WORKDIR /app


Inside the Docker container, the working directory will be /app. All files will be copied into this directory.

Next, the Docker file installs system dependencies. It updates the system, installs essential libraries (like curl), and removes unnecessary files to save space. These steps are optional but recommended for production-grade projects.

COPY . /app


This line copies all contents from the local directory into the Docker container’s /app directory.

To install Python dependencies, the Docker file runs:

RUN pip install --no-cache-dir -e .


This installs all required libraries specified in setup.py.

--no-cache-dir ensures that no previous cache is used, which guarantees a fresh installation.

Finally, the Docker file exposes the ports that the app uses:

9999 for the backend (FastAPI)

8501 for the frontend (Streamlit)

The command to run the app inside the Docker container is the same as running it locally:

CMD ["python", "app/main.py"]


Make sure the ports are exposed; otherwise, the app will not work.

With this, the Docker file setup is complete. This Dockerized project can now be deployed to any platform, ensuring consistency between local development and production environments.

**Summary:**

(i) Purpose

Create a Docker image for the project to ensure consistent deployment across environments.

Docker encapsulates all dependencies, Python version, and project files.

(ii) Create Dockerfile

In the root directory, create a file named Dockerfile.

You can copy the code from GitHub or follow the instructor’s explanation.

(iii) Specify Base Image

FROM python:3.11-slim


Uses Python 3.11 slim as a lightweight, stable base.

Avoid unstable versions like Python 3.13.

(iv) Set Environment Variables

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files.

PYTHONUNBUFFERED=1: Ensures real-time logging for debugging and monitoring.

(v) Set Working Directory

WORKDIR /app


Inside the container, /app will be the working directory.

All project files are copied here.

(vi) Copy Project Files

COPY . /app


Copies all local project files into the Docker container’s /app directory.

(vii) Install Dependencies

RUN pip install --no-cache-dir -e .


Installs all Python dependencies specified in setup.py.

--no-cache-dir ensures a fresh install without using cached packages.

(viii) Optional System Dependencies

You can update the system and install essential libraries (like curl).

Clean up unnecessary files to reduce Docker image size (recommended for production).

(ix) Expose Ports

EXPOSE 9999  # Backend FastAPI
EXPOSE 8501  # Frontend Streamlit


Expose ports so the app is accessible outside the container.

9999 → FastAPI backend, 8501 → Streamlit frontend.

(x) Run the App

CMD ["python", "app/main.py"]


Starts the app inside the container, same as running locally with Python.

**What we did:**

(i) Docker ensures environment consistency for local development and production.

(ii) All dependencies, ports, and project structure are encapsulated in a single container.

(iii) This Dockerized setup allows deployment on any platform without environment conflicts.

**10. Jenkins Setup for CI-CD Deployment**

In this video, we start the CI/CD deployment for our project, beginning with the Jenkins setup. First, in the root directory, you create a folder named custom_jenkins and inside it, a Dockerfile. You can copy the Dockerfile content from the GitHub link provided. This Dockerfile will build the Jenkins Docker image and run the Jenkins container. The setup uses a dependent Docker approach, meaning one container runs Jenkins, and inside it, Docker can run other containers if needed.

Next, open your Ubuntu WSL terminal in VS Code and navigate to the custom Jenkins folder using cd custom_jenkins. Then, build the Jenkins Docker image using the command provided. You might see some warnings in red, but these can be safely ignored. Once the image is built, run the Jenkins container. You will receive an alphanumeric key indicating that the container is running. You can also verify it by running docker ps. The Jenkins service will run on port 8080 as defined in the Dockerfile.

To log in to Jenkins, you need the initial admin password. Retrieve it using docker logs <jenkins_container_name> and copy it carefully, as it is generated only once. Next, get the IP address of your container, then open a browser and navigate to http://<container_ip>:8080. Enter the initial admin password and continue with the setup, choosing “Install suggested plugins” to automatically install useful plugins for CI/CD.

After Jenkins is accessible, you need to install Python inside the Jenkins container to run Python commands like python app.py. Open the Jenkins container terminal and update packages using apt-get update, then install Python 3 with apt-get install python3. Check the installation with python3 --version. To use Python 3 as the default python command, run the provided alias command. Optionally, install pip if needed.

Finally, complete the Jenkins setup wizard by creating a username, password, full name, and email. Save and finish the setup, then restart the Jenkins container using docker restart <jenkins_container_name> to apply the changes. After restarting, log in again using your credentials. At this point, Jenkins is fully set up, and the dashboard is ready. In the next video, we will start creating CI/CD pipelines.

**Summary:**

Jenkins Setup for CI/CD Deployment

(i) Create Jenkins Folder and Dockerfile

In the root directory, create a folder named custom_jenkins.

Inside this folder, create a Dockerfile.

Copy the Dockerfile content from the GitHub repository.

This Dockerfile builds a Jenkins Docker image and runs a Jenkins container.

Dependent Docker approach: Jenkins container can run other Docker containers if required.

(ii) Build Jenkins Docker Image

Open Ubuntu WSL terminal in VS Code.

Navigate to the Jenkins folder:

cd custom_jenkins


Build the Docker image using the command provided in the video or GitHub.

Note: Some red warnings may appear; they can be ignored safely.

(iii) Run Jenkins Container

Run the container from the built image.

After running, you will receive an alphanumeric key indicating the container is running.

Verify the container is running:

docker ps


Jenkins service will be available on port 8080 as defined in the Dockerfile.

(iv) Log in to Jenkins

Retrieve the initial admin password:

docker logs <jenkins_container_name>


Copy the password carefully—it is generated only once.

Get the container IP address, then open a browser:

http://<container_ip>:8080


Enter the initial admin password.

Continue the setup and select “Install suggested plugins” to install essential CI/CD plugins automatically.

(v) Install Python in Jenkins Container

Open the terminal inside the Jenkins container.

Update packages:

apt-get update


Install Python 3:

apt-get install python3


Check installation:

python3 --version


To make Python 3 the default python command, run the alias command provided in the video.

Optionally, install pip if needed.

(vi) Complete Jenkins Setup Wizard

Create a username, password, full name, and email.

Save and finish the setup.

Restart the Jenkins container to apply changes:

docker restart <jenkins_container_name>


Log in again with your credentials.

(vii) Result

Jenkins is now fully set up.

**What we did:**

(i) Jenkins runs inside a Docker container on port 8080.

(ii) Initial admin password is generated only once—save it carefully.

(iii) Installing Python inside the container allows Jenkins to run Python commands like python app.py.

(iv) Using suggested plugins ensures essential CI/CD features are ready.

**11. GitHub Integration with Jenkins**

In this video, we focus on integrating GitHub with Jenkins. Start by opening your GitHub account, navigating to your profile settings, and going to Developer Settings. Under Developer Settings, select Personal Access Tokens and create a classic token. Use your GitHub password for verification, give your token a name (e.g., the project name), set an expiration date, and assign the required permissions: repo and repo:hook. Copy this token carefully and keep the tab open because it cannot be retrieved again once closed.

Next, open Jenkins and navigate to Manage Jenkins → Credentials → Global credentials → Add credentials. Choose "Username with password," set your GitHub username as the username, and the previously generated personal access token as the password. Give the credential a meaningful ID and description, such as GitHub token, and save it. This token allows Jenkins to access your GitHub repositories based on the permissions you provided.

After setting up the credentials, return to the Jenkins dashboard and create a new item for your project. Name it appropriately (e.g., multi AI agent) and select "Pipeline" as the project type. In the pipeline configuration, choose "Pipeline script from SCM," select Git as the source, and paste your GitHub repository URL. In the credentials section, select the GitHub token you created earlier. Make sure to set the branch to main and keep the script path as Jenkinsfile. Apply and save the configuration.

Next, generate the pipeline script by opening Pipeline Syntax → checkout. Paste your repository URL, select the credentials, set the branch to main, and generate the script. Keep this code handy for later. Then, in VS Code, create a Jenkinsfile in the root directory. Copy the content from the provided GitHub resource, but comment out the SonarQube analysis stages using Ctrl + /. The first stage in your pipeline is cloning the GitHub repository into Jenkins. Ensure that all other stages are commented out initially.

Once the Jenkinsfile is ready, push it to GitHub using the standard Git commands: git add ., git commit -m "commit", and git push origin main. After updating the repository, return to Jenkins, open your pipeline, and click Build Now. If the build fails, check the console output. Often, failures occur due to unset environment variables. Comment out the environment block, push the changes again, and rebuild. Once successful, the build console will show a green tick, indicating that the GitHub repository has been successfully cloned into the Jenkins workspace.

You can verify this by going to the workspace in Jenkins, where all the files from your GitHub repository—including Dockerfile, Jenkinsfile, requirements, app code, and front-end/back-end directories—will now be available. This completes the GitHub-Jenkins integration. In the next video, we will move on to the subsequent steps of the CI/CD pipeline.

**Summary:**

(i) Generate GitHub Personal Access Token (PAT)

Open GitHub → Profile → Settings → Developer Settings → Personal Access Tokens → Tokens (classic).

Click Generate new token → Classic token.

Verify with your GitHub password.

Provide a token name (e.g., project name) and set an expiration date.

Assign the required permissions:

repo

repo:hook

Copy the token carefully and keep the tab open, as it cannot be retrieved later.

(ii) Add Credentials in Jenkins

Open Jenkins → Manage Jenkins → Credentials → Global credentials → Add Credentials.

Choose “Username with password”.

Set your GitHub username as the username.

Paste the personal access token as the password.

Give it a meaningful ID and description (e.g., GitHub token).

Save the credentials.

This allows Jenkins to access your GitHub repository using the permissions provided.

(iii) Create Jenkins Pipeline for Project

Return to Jenkins dashboard → New Item.

Name the project (e.g., multi AI agent) and select Pipeline as the project type.

Configure the pipeline:

Pipeline script from SCM

Source: Git

Repository URL: Paste your GitHub repo URL

Credentials: Select the GitHub token created earlier

Branch: main

Script path: Jenkinsfile

Apply and save.

(iv) Generate Pipeline Script

In Jenkins, open Pipeline Syntax → checkout.

Paste your repository URL, select credentials, set branch to main, and generate the script.

Keep this script handy for your Jenkinsfile.

(v) Create Jenkinsfile in VS Code

In the project root, create a Jenkinsfile.

Copy the content from the provided GitHub resource.

Comment out SonarQube analysis stages (Ctrl + /) for initial setup.

Ensure the first stage clones the GitHub repository.

Keep other stages commented initially.

(vi) Push Jenkinsfile to GitHub

git add .
git commit -m "Add Jenkinsfile"
git push origin main


(vii) Build Pipeline in Jenkins

Go back to Jenkins → Open your pipeline → Click Build Now.

If the build fails:

Check console output for errors (often due to unset environment variables).

Comment out the environment block in Jenkinsfile.

Push changes and rebuild.

Once successful, a green tick indicates the GitHub repo has been cloned into Jenkins workspace.

(viii) Verify Workspace

Go to the workspace in Jenkins.

All files from your GitHub repository should be available:

Dockerfile

Jenkinsfile

requirements.txt

app code

front-end and back-end directories

**What we did:**

(i) The personal access token allows secure communication between Jenkins and GitHub.

(ii) Commenting out certain stages initially ensures the first build is successful.

(iii) Workspace verification confirms that the GitHub repository is fully cloned.

**12. SonarQube Integration with Jenkins**

In this video, we will integrate SonarQube with Jenkins. First, open a browser and search for Docker Hub. In Docker Hub, search for sonarqube and open the relevant page. Scroll down to find the Linux commands for running SonarQube using Docker. Copy these commands one by one and run them in a new terminal or WSL instance. These commands will set up SonarQube on your machine, and it will run on port 9000. If the image is not found locally, Docker will pull it from the SonarQube repository. Once the setup is complete, you can verify the installation using docker ps to see the SonarQube container running alongside Jenkins.

Next, open SonarQube in a browser using the IP address you used for Jenkins but replace the port with 9000. Log in with the default credentials (admin/admin) and update the password if needed. Then, go back to Jenkins and install the necessary plugins: SonarQube Scanner and SonarQube Quality Gates. After installation, restart Jenkins to apply the changes.

In SonarQube, create a new local project for your pipeline, providing a project name (e.g., LMops) and keeping the branch as main. Generate an access token for Jenkins to authenticate with SonarQube. Copy this token and add it to Jenkins credentials as a Secret Text type. Next, configure Jenkins to recognize the SonarQube server by adding the server details, including the token and the SonarQube URL, under Manage Jenkins → System → SonarQube servers.

After that, configure the SonarQube Scanner tool in Jenkins under Manage Jenkins → Tools. Give it a name (e.g., SonarQube) and enable automatic installation with the default version. Once the scanner is set up, open your Jenkinsfile in VS Code, uncomment the environment variable and the SonarQube analysis stage, and replace the placeholders with your actual SonarQube token, project name, tool name, and environment name. Push the updated Jenkinsfile to GitHub.

Before running the pipeline, ensure that both Jenkins and SonarQube are on the same Docker network. If not, create a new Docker network and connect both containers to it. This allows Jenkins to communicate with SonarQube. Once the network is configured, build the pipeline in Jenkins. If successful, the console output will indicate that the SonarQube analysis has started and completed.

Finally, open SonarQube and check the project dashboard. You will see metrics such as maintainability, coverage, code issues, security hotspots, reliability, duplications, and overall quality gate status. The analysis provides suggestions for improving your code quality and ensures that your project follows production-grade standards. This completes the integration of SonarQube with Jenkins. Previously, we integrated GitHub with Jenkins, and now SonarQube has been successfully integrated as well.

**Summary:**

(i) Set Up SonarQube Using Docker

Open Docker Hub in a browser and search for SonarQube.

Copy the Linux commands for running SonarQube with Docker.

Run them in a terminal or WSL.

SonarQube will run on port 9000.

If the image is not available locally, Docker will pull it automatically.

Verify the container is running using:

docker ps


(ii) Access SonarQube

Open a browser and navigate to:

http://<your-docker-ip>:9000


Log in with default credentials: admin/admin.

Update the password as needed.

(iii) Install Jenkins Plugins

Go to Jenkins → Manage Jenkins → Manage Plugins → Available.

Install the following plugins:

SonarQube Scanner

SonarQube Quality Gates

Restart Jenkins to apply the changes.

(iv) Create a SonarQube Project

In SonarQube, create a new local project.

Provide a project name (e.g., LMops) and set the branch to main.

Generate an access token for Jenkins to authenticate with SonarQube.

(v) Add SonarQube Credentials in Jenkins

Go to Jenkins → Manage Jenkins → Credentials → Global → Add Credentials.

Select Secret Text type.

Paste the SonarQube token and save.

(vi) Configure SonarQube Server in Jenkins

Go to Manage Jenkins → System → SonarQube Servers.

Add the server details:

Server Name

SonarQube URL (e.g., http://<docker-ip>:9000)

Token (Secret Text credential added earlier)

(vii) Configure SonarQube Scanner in Jenkins

Go to Manage Jenkins → Tools → SonarQube Scanner.

Give it a name (e.g., SonarQube)

Enable automatic installation with the default version.

(viii) Update Jenkinsfile

Open Jenkinsfile in VS Code.

Uncomment the environment variables and SonarQube analysis stage.

Replace placeholders with:

SonarQube token

Project name

Tool name

Environment name

Push the updated Jenkinsfile to GitHub.

(ix) Ensure Jenkins and SonarQube Are on the Same Docker Network

If not, create a new Docker network:

docker network create <network_name>


Connect both Jenkins and SonarQube containers to this network:

docker network connect <network_name> <container_name>


(x) Run the Pipeline

Open your Jenkins pipeline and click Build Now.

Console output will show SonarQube analysis starting and completing.

(xi) Verify SonarQube Analysis

Open the SonarQube project dashboard.

Metrics displayed include:

Maintainability

Coverage

Code issues

Security hotspots

Reliability

Duplications

Overall quality gate status

Analysis provides suggestions to improve code quality and ensures production-grade standards.

**What we did:**

(i) SonarQube runs on port 9000; Jenkins on 8080.

(ii) Both containers must be on the same Docker network for communication.

(iii) SonarQube token ensures secure authentication from Jenkins.

(iv) Once integrated, code quality metrics are automatically analyzed in the CI/CD pipeline.

**13. Build & Push Image to AWS ECR**

In this video, we will create the build and push stage of our CI/CD pipeline. The goal is to build a Docker image for our project and push it to AWS Elastic Container Registry (ECR), which is a platform to store Docker images. Docker images cannot be stored just anywhere; they need a container registry like ECR.

First, open Jenkins and go to Manage Jenkins → Plugins. In the available plugins, search for and install AWS Credentials and AWS SDK plugins. After the plugins are installed, we need to install the AWS CLI inside the Jenkins container. Open the terminal of the Jenkins container and run the commands to update packages (apt update), install curl and unzip, and finally install AWS CLI. Verify the installation by running aws --version, which should display the installed AWS CLI version. Once installed, exit the container and restart Jenkins using docker restart <container-name> to apply the changes.

Next, log in to your AWS account and create a new IAM user for Jenkins. Give this user a name, attach the Amazon EC2 Container Registry (ECR) Full Access policy, and create the user. Generate an access key for this user, which will be used for Jenkins authentication. In Jenkins, go to Manage Jenkins → Credentials → Global → Add Credentials and add a new AWS credential using the generated access key and secret key. Give it an ID such as AWS token for reference.

After that, create an ECR repository in AWS where your Docker images will be stored. Provide a name for the repository (e.g., my-repo) and leave other settings as default. In VS Code, open your Jenkinsfile and uncomment the build and push stage, along with the associated environment variables. Update the AWS region, ECR repository name, and image tag to match your AWS setup. Also, set the credential ID to match the AWS token you created in Jenkins.

The build and push stage in the pipeline script performs four main tasks: it logs into AWS ECR, builds the Docker image using your project’s Dockerfile, tags the image with the latest tag, and pushes it to the specified ECR repository. After making these updates, push the changes to GitHub. Then, go to your Jenkins dashboard and build the pipeline. This stage may take some time, as it involves building the Docker image and pushing it to AWS ECR.

Once the pipeline completes successfully, check the console output for a green tick. To ensure the build and push were successful, go to your ECR repository in AWS. You should see the Docker image listed with the latest tag, including details like size and creation time. If the image appears correctly, it confirms that the build and push stage was completed successfully. With this stage done, our project is now ready for the next step in the CI/CD workflow.

**Summary:**

(i) Install Required Jenkins Plugins

Open Jenkins → Manage Jenkins → Manage Plugins.

Search for and install:

AWS Credentials

AWS SDK

(ii) Install AWS CLI in Jenkins Container

Open the terminal inside the Jenkins container.

Update packages and install dependencies:

apt update
apt install curl unzip -y


Install AWS CLI:

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


Verify installation:

aws --version


Exit the container and restart Jenkins:

docker restart <container-name>


(iii) Create AWS IAM User for Jenkins

Log in to your AWS account.

Create a new IAM user for Jenkins.

Attach Amazon EC2 Container Registry (ECR) Full Access policy.

Generate Access Key ID and Secret Access Key.

(iv) Add AWS Credentials in Jenkins

Go to Jenkins → Manage Jenkins → Credentials → Global → Add Credentials.

Select AWS Credentials.

Enter the Access Key ID and Secret Key from AWS.

Give it an ID such as AWS token.

(v) Create ECR Repository in AWS

In AWS console, navigate to ECR → Create Repository.

Provide a repository name (e.g., my-repo) and leave other settings as default.

(vi) Update Jenkinsfile

Open Jenkinsfile in VS Code.

Uncomment the build and push stage along with its environment variables.

Update parameters:

AWS region

ECR repository name

Docker image tag

Credential ID (the AWS token created in Jenkins)

(vii) Understand the Build and Push Stage
The pipeline stage performs four main tasks:

Login to AWS ECR using AWS CLI and Jenkins credentials.

Build the Docker image using the project’s Dockerfile.

Tag the image with the latest tag (or custom tag).

Push the Docker image to the specified ECR repository.

(viii) Push Changes and Trigger Jenkins Pipeline

Push the updated Jenkinsfile to GitHub:

git add Jenkinsfile
git commit -m "Enable build and push stage"
git push origin main


In Jenkins, open the pipeline and click Build Now.

This stage may take a few minutes as it builds and pushes the Docker image.

(ix) Verify Success

Check Jenkins console output for a green tick indicating a successful build.

Open your AWS ECR repository.

Verify that the Docker image appears with the correct tag, size, and creation time.

**What we did:**

(i) AWS ECR is required as the container registry.

(ii) Jenkins must have AWS credentials configured to push images.

(iii) The Docker image must be built and tagged correctly before pushing.

(iv) Successful completion ensures the project is ready for the next CI/CD stage.

**14. Deployment to AWS Fargate**

In this final video of the CI/CD deployment series, we deploy our project from AWS ECR to AWS ECS using Fargate. Start by opening the ECS (Elastic Container Service) console in AWS. First, create a new cluster by navigating to Clusters → Create Cluster, selecting AWS Fargate as the deployment type, and giving it a name, e.g., multi AI agent cluster. Fargate provides a serverless deployment option, so you don’t need to manage any servers. Once the cluster creation is initiated, it may take a few minutes to complete.

Next, create a task definition by navigating to Task Definitions → Create New Task Definition, selecting Fargate, and naming it (e.g., multi AI agent def). Choose Linux as the operating system, assign two CPUs and 6 GB of memory. Within the task definition, define a container, give it a name, and provide the image URI from your ECR repository. Mark the container as essential and set up port mappings for your applications—typically port 8501 for Streamlit and 9999 for FastAPI. Then, add any required environment variables from your .env file, such as API keys, because sensitive data is not pushed to GitHub. Once configured, create the task definition.

After the task definition, navigate back to your cluster and create a service. Select the task definition created earlier and ensure it uses the latest revision. Assign a service name, enable public IP in the networking section, and keep other settings as default. The service creation may take 5–10 minutes. After the service is created, configure security groups to allow inbound traffic on the required ports (8501 and 9999) so that your applications are accessible publicly. Ensure the service is associated with the correct security group.

Once the service is deployed, you can access your application using the public IP of the running task along with the appropriate port (e.g., http://<public-ip>:8501 for Streamlit). Initially, this is considered a manual deployment, as it uses a specific Docker image from ECR. Any new changes in the code will not reflect until the CI/CD pipeline is executed. By uncommenting and updating the final deployment stage in your Jenkinsfile with the correct cluster and service names, you can push the updated code to GitHub, trigger the Jenkins pipeline, and deploy the latest image automatically.

During deployment, you might encounter an Access Denied error if the IAM user lacks the necessary ECS permissions. To fix this, attach the Amazon ECS Full Access policy to your Jenkins IAM user. After this, rebuild the Jenkins pipeline, and the new image will be deployed. Note that updates are not instant; the new task is initially pending while the previous task continues running. After a few minutes, the old task shuts down, and the new task becomes active, ensuring the latest changes are live.

With this setup, the CI/CD pipeline ensures that every time a new image is built and pushed to ECR, ECS automatically deploys it using the latest image. However, automatic triggering of the pipeline on GitHub pushes requires Jenkins to be online, either on an EC2 instance or a similar accessible platform, because webhooks cannot connect a local Jenkins instance to GitHub. Temporary solutions like Ngrok can expose a local Jenkins to the internet, but for a full production-ready CI/CD setup, an online Jenkins instance is recommended. This completes the CI/CD deployment pipeline for the Multi-AI Agent project, combining GitHub integration, Docker build & push, and ECS Fargate deployment.

**Summary:**

(i) Create an ECS Cluster

Open the ECS Console → Clusters → Create Cluster.

Select AWS Fargate as the deployment type (serverless, no need to manage servers).

Give it a name, e.g., multi AI agent cluster.

Wait a few minutes for the cluster to be created.

(ii) Create a Task Definition

Navigate to Task Definitions → Create New Task Definition → Fargate.

Give it a name, e.g., multi AI agent def.

Configure:

OS: Linux

CPU: 2

Memory: 6 GB

Add a container:

Name the container.

Provide the ECR image URI (from the build & push stage).

Mark the container as essential.

Set port mappings: 8501 for Streamlit, 9999 for FastAPI.

Add environment variables from your .env file (API keys, etc.).

Create the task definition.

(iii) Create a Service

Go back to your cluster → Create Service.

Select the task definition (latest revision).

Provide a service name.

Enable public IP in networking.

Keep other settings as default.

Service creation may take 5–10 minutes.

(iv) Configure Security Groups

Allow inbound traffic for required ports:

8501 → Streamlit front-end

9999 → FastAPI back-end

Ensure the service is associated with the correct security group.

(v) Access the Application

Use the public IP of the running ECS task:

Streamlit: http://<public-ip>:8501

FastAPI: http://<public-ip>:9999

Initially, this is a manual deployment using a specific Docker image from ECR.

(vi) Enable Automated CI/CD Deployment

Uncomment and update the final deployment stage in your Jenkinsfile.

Provide correct cluster name, service name, and other ECS details.

Push updated Jenkinsfile to GitHub → trigger Jenkins pipeline → deploy latest image automatically.

(vii) Common Issues & Fixes

Access Denied Error:

Ensure the Jenkins IAM user has Amazon ECS Full Access policy.

Rebuild the Jenkins pipeline after updating IAM permissions.

Task Update Delay:

ECS launches the new task while the old task is still running.

After a few minutes, the old task stops, and the new task becomes active.

(viii) Notes on Production-Ready Deployment

Automatic pipeline triggering requires Jenkins to be accessible online.

Local Jenkins cannot receive GitHub webhooks directly; temporary solutions like Ngrok can expose it.

For production, host Jenkins on EC2 or another publicly accessible server.

**What we did:**

(i) Every time a new Docker image is built and pushed to ECR, ECS Fargate deploys it using the latest image.

(ii) The CI/CD pipeline now covers:

(a) GitHub integration

(b) Docker build & push to ECR

(c) Deployment to ECS Fargate

**15. Cleanup Process:**

This video explains the cleanup process after completing your CI/CD deployment to avoid unnecessary charges and free up system resources. First, navigate to your ECS clusters. Running clusters continuously can lead to significant charges, so it’s important to delete them once you’re done. Open your cluster, click Delete Cluster, confirm by copying and pasting the cluster name, and the cluster along with its services will be automatically removed.

Next, go to your ECR (Elastic Container Registry) repository and delete the repository to remove all stored Docker images. This prevents extra storage costs. Optionally, you can also delete your IAM user, though it does not incur charges; keeping it is a personal choice.

Additionally, it is a good practice to clean up your local Docker environment to avoid conflicts with future projects, especially if your Docker images or project names are similar. Open your Ubuntu terminal and run the command docker system prune -a -f. This will delete all unused images, containers, volumes, and caches, freeing up space on your system. For example, this command may reclaim several gigabytes of disk space.

Finally, regarding the dynamic IP issue when deploying tasks to ECS: by default, each task may receive a different public IP. If you prefer a fixed IP, you can enable load balancing while creating your ECS service. This will route traffic through a static IP, ensuring consistency for accessing your application. However, for this project, using a dynamic IP is sufficient, as the main focus is learning how to set up a complete CI/CD deployment.

This concludes the cleanup process. Following these steps ensures you avoid unnecessary AWS charges, free up local system resources, and maintain a clean environment for future projects.

**Summary:**

(i) Delete ECS Clusters

Navigate to ECS → Clusters.

Running clusters incur costs, so delete them after use.

Open the cluster → Delete Cluster → confirm by copying and pasting the cluster name.

This removes the cluster along with all associated services.

(ii) Delete ECR Repositories

Go to ECR (Elastic Container Registry) → select the repository.

Delete the repository to remove all stored Docker images.

This prevents additional storage charges.

(iii) Optional: Delete IAM User

Deleting the IAM user is optional as it does not incur charges.

Keeping it depends on your future access needs.

(iv) Clean Local Docker Environment

To avoid conflicts in future projects, clean unused Docker resources.

Open Ubuntu terminal and run:

docker system prune -a -f


This removes:

Unused images

Stopped containers

Volumes

Build caches

Frees up disk space (several GBs can be reclaimed).

(v) Dynamic vs Fixed IP for ECS Tasks

By default, ECS tasks receive dynamic public IPs.

For a static IP: enable load balancing when creating the ECS service.

Dynamic IP is sufficient for learning and testing purposes.

**What we did:**

(i) Avoid unnecessary AWS charges.

(ii) Free up local system resources.

(iii) Maintain a clean environment for future projects.
