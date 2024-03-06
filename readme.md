### Purpose
This app allows the user to query the data that they specify. The benefit of this is new data (after LLM was trained ~ 2022) can be queried as well.
The app uses Gemini-Pro 1.0, however it should be straight-forward to opt it out for another LLM in the code.


### Create Virtual Environment (Recommended to use Conda Environment)

```
- clone this repository to a local directory
- conda create -n "env_name" python=3.9.15
- activate environment then cd into this directory
- pip install -r requirements.txt
```

### Caution
Google Gemini Pro currently has location restrictions. Refer to https://ai.google.dev/available_regions to see if your region is supported.
If not, consider using a VPN or switch the embeddings and llm initialization with another desired model.
```python
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)
```

Store your Google API Key in a .env file using keyword: "GOOGLE_API_KEY"

### HOW TO RUN

#### Local
```
Download/Clone the repository and create a .env file with your (free) GOOGLE_API_KEY
Create a virtual environment (Python 3.9.15) and install depencies using: pip install -r app/requirements.txt
From Parent Directory, Run: streamlit run app/app_v6.py
(Default): Go to http://localhost:8501/
```

#### Using Docker
```
Run Command from same directory as Dockerfile: docker build -t <name_of_image> .
Check Image is Created: docker images
Run App: docker run -p 8501:8501 <name_of_image>
```
Note that since the "GOOGLE_API_KEY" needs to be set in a .env file, it is suggested that docker-compose also be utilized.
More information can be found here: https://docs.docker.com/compose/environment-variables/set-environment-variables/

#### Streamlit Cloud
```
Go To: https://chat-w-data.streamlit.app/

CAUTION: Currently, all data sources currently work on cloud except MySQL. Since the implementation of MySQL would use a local instance, it's 
not possible to connect to localhost. 
```