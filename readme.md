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

Store your Google API Key in a .env folder using keyword: "GOOGLE_API_KEY"