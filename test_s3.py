import boto3
import pandas
from PIL import Image

# Creating the low level functional client
client = boto3.client(
    's3',
    aws_access_key_id='AKIAQF3NP3LVMOUHVKFU',
    aws_secret_access_key='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',
    region_name='us-east-2'
)

# Creating the high level object oriented interface
resource = boto3.resource(
    's3',
    aws_access_key_id='AKIAQF3NP3LVMOUHVKFU',
    aws_secret_access_key='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',
    region_name='us-east-2'
)

# Fetch the list of existing buckets
clientResponse = client.list_buckets()

# Print the bucket names one by one
print('Printing bucket names...')
for bucket in clientResponse['Buckets']:
    print(f'Bucket Name: {bucket["Name"]}')

# Create the S3 object
obj = client.get_object(
    Bucket='tdistaticfiles',
    Key='image/amaizeing.png'
)

# Read data from the S3 object
# data = pandas.read_csv(obj['Body'])
# print(data)
img = Image.open(obj['Body'])
img.show()
print()