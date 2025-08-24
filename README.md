# LLMOps-and-AIOps-Skills

**(Cloud CI/CD Toolkit like Gitlab; Jenkins is a Local**

### **We are converting here; Why not RGB to Gray; Because when we get a Image from Internet it is BGR and not RGB** gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### Next is temperature; Higher the temperature, more creative our model is; More creative more hallucinative, if suppose we have given Virat Kohli, it can detect as other clebrity, so give it in between 0.2-0.5

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
