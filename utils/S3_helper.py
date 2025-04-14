import os
from dotenv import load_dotenv
import boto3


load_dotenv()


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')  # fallback if not in .env


s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

response = s3.list_buckets()
print("Buckets:")
for bucket in response['Buckets']:
    print(f"  - {bucket['Name']}")
