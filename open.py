import base64
import requests
import datetime
from decouple import config
# Path to the image you want to send
image_path = 'captured_frames/test.jpeg'

# API endpoint and API key
url = "https://detect.roboflow.com/shoplifting-detection-oxvwp/1"
params = {
    'api_key': config('roboflow_api_key')
}


# params = {
#     "api_key": "fH57K9cUP8mxPIvT51tY",
#     "image": "https://source.roboflow.com/SIq2BvVeimbJs5zcmYfuqXZq7TJ2/brJAZqt6fWYJnUXJ7JXw/original.jpg"
# }



# Read the image and encode it as base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


print(type(encoded_image))


headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}

# Make the POST request with the base64-encoded image as the body
strt = datetime.datetime.now()

response = requests.post(url, params=params, data=encoded_image, headers=headers)

end = datetime.datetime.now() - strt

print(end.seconds)
# response = requests.post(url, params=params)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response Body:", response.json())
