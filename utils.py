from ultralytics import YOLO
import numpy as np
import boto3
import os


def download_image_from_s3(bucket_name, s3_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, local_file_path)
        print(f"Image downloaded successfully from S3")
        return True
    except Exception as e:
        print(f"Error downloading image from S3: {e}")
        return False


def login_to_aws(S3_ACESS_KEY_ID, S3_SECRET_ACESS_KEY):
    try:
        # Create a session using your credentials
        session = boto3.Session(
            aws_access_key_id=S3_ACESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACESS_KEY
        )
        print("Login successful!")
        return session
    except Exception as e:
        print(f"Error logging in to AWS: {e}")
        return None


def load_to_S3(img: str, bucket_name: str, s3_key: str):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(img, bucket_name, s3_key)
        print(f"Image uploaded sucessfully")
    except Exception as e:
        print(f"Error uploading image to S3 {e}")


def handle_task_backbone(task_type: str, img: np.array, img_name: str):
    output_dir = os.getcwd() + "/output/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if task_type == "detection":
        model = YOLO("yolov8n.pt")
        results = model([img])
        filename = output_dir + img_name
        bucket_name = "apprunnerdetections"
        for r in results:
            r.save(filename=filename)
        load_to_S3(filename, bucket_name, img_name)
        return True
    elif task_type == "segmentation":
        model = YOLO("yolov8n-seg.pt")
        results = model([img])
        filename = output_dir + img_name
        bucket_name = "apprunnersegmentations"
        for r in results:
            r.save(filename=filename)
        load_to_S3(filename, bucket_name, img_name)
        return True
    else:
        return False
