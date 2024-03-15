from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from schema import TaskRequest
import cv2
from utils import handle_task_backbone, login_to_aws, download_image_from_s3
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY')


app.add_middleware(
    CORSMiddleware,
    # Update this with your frontend origin(s) or "*" for all origins
    allow_origins=["*"],
    allow_credentials=True,
    # Specify the HTTP methods you want to allow
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],  # Specify the HTTP headers you want to allow
)


@app.post("/task/")
def handle_task(task: TaskRequest):
    bucket_name = "apprunnerimages"

    task_type = task.task
    img_name = task.imgPath
    login_to_aws(S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)
    download_image_from_s3(bucket_name, img_name, img_name)
    img = cv2.imread(img_name)
    result = handle_task_backbone(task_type, img, img_name)
    if result:
        return {"Task Completed Sucessfully"}
    else:
        return HTTPException(status_code=400, detail="Task type not available")


if __name__ == "__main__":
    app()
