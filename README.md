# LLMOps-and-AIOps-Skills

**To Trigger setup.py - pip install -e .**

**(Cloud CI/CD Toolkit like Gitlab; Jenkins is a Local**

### **We are converting here; Why not RGB to Gray; Because when we get a Image from Internet it is BGR and not RGB** gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### Next is temperature; Higher the temperature, more creative our model is; More creative more hallucinative, if suppose we have given Virat Kohli, it can detect as other clebrity, so give it in between 0.2-0.5

**A. AI Anime Recommender Project**

**E. Celebrity Detector and QnA**

**F. AI Music Composer**

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

**4. Main Application code using Streamlit**

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
