from fastapi import FastAPI, File, UploadFile,Request,Response
import subprocess
from PIL import Image
import io
import os
from fastapi.templating import Jinja2Templates
from azure.storage.blob import BlobServiceClient




app = FastAPI()
templates=Jinja2Templates(directory="templates")
my_connection_string='DefaultEndpointsProtocol=https;AccountName=aiworkspace4684782811;AccountKey=+IRiIf1QCRTNc9qDhJRQdoqmTH43tdMOXwSQxFXGKrzspKIa65JJwSo63wHa/mbfxF6t5+vbdZno+AStN6rTfw==;EndpointSuffix=core.windows.net'


def get_image_data_from_blob():
    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(my_connection_string)

    # Get a reference to the blob container
    container_client = blob_service_client.get_container_client("azureml")

    # Get a reference to the blob
    blob_client = container_client.get_blob_client("myimage.jpg")

    # Download the blob data
    blob_data = blob_client.download_blob().readall()

    # Return the blob data as a bytes object
    return blob_data

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/uploads/")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Save the image to a temporary file
    temp_file_path = 'temp_img.jpg'
    image.save(temp_file_path)
    
    # Call the predict function using subprocess.Popen
    results=subprocess.Popen(["python", "predictWithOCR.py", "model='Licence_Plate_Repo/best.pt'", f"source={temp_file_path}"], stdout=subprocess.PIPE)
    output, error = results.communicate()
    
    # Delete the temporary file
    os.remove(temp_file_path)
    i= get_image_data_from_blob()
    
   # file_path = "runs/detect/train/temp_img.jpg"  # replace with your actual file path
    # Return the output
    return Response(content=i, media_type='image/jpeg')

def run():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()

 
