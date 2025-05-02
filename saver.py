from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
import shutil


load_dotenv()
TOKEN = os.environ['HF_TOKEN']

api = HfApi(token=TOKEN)


# api.delete_files(
#     repo_id='DC-Uni/LLM_NLP_HW_1',
#     repo_type='model',
#     delete_patterns=os.listdir('pretrained_models')
# )

api.upload_folder(
    folder_path='./pretrained_models',
    repo_id='DC-Uni/LLM_NLP_HW_1',
    repo_type='model'
)