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


@app.post("/task/")
def handle_task(task: TaskRequest):
    bucket_name = "apprunnerimages"

    task_type = task.task
    img_path = task.imgPath
    img_name = img_path.split("/")[-1]  # may change
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
