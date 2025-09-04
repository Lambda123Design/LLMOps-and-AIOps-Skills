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
