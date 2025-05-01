from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.environ['HF_TOKEN']

api = HfApi(token=TOKEN)

api.upload_folder(
    folder_path='./pretrained_models',
    repo_id='DC-Uni/LLM_NLP_HW_1',
    repo_type='model'
)