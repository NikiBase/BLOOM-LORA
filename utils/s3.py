import os
import tempfile
import zipfile

import boto3


def zip_n_store(directory_path, bucket_name, s3_key):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "tmp.zip")

        with zipfile.ZipFile(temp_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(
                        file_path, arcname=os.path.relpath(file_path, directory_path)
                    )

        s3 = boto3.client("s3")
        s3.upload_file(temp_file, bucket_name, s3_key)

