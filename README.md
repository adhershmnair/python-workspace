# Python Workspace files.

Install Packages from a `requirements.txt` File

```
pip install -r requirements.txt
```

## facerecognition.py

Basic face recognition using known images in `images` folder.
Add known_images to `images` folder with proper naming, so that it will recongnize the face based on the images.


## facerecognition-google-photos.py

Integrated Google Photos to identify the images. To get images from Google Photos, you would need to use the Google Photos API.
You need to ensure that the redirect URI in your Google Cloud Console matches the one in your code. Here's how to do it:

1. Go to the Google Cloud Console (console.cloud.google.com).
2. Select your project.
3. Go to "APIs & Services" > "Credentials".
4. Click on the OAuth 2.0 Client ID that you're using.
5. Under "Authorized redirect URIs", add the redirect URI that your application is using. If you're running the application locally, it's likely http://localhost:8080/. The exact URI depends on the port you're using in your code (flow.run_local_server(port=8080)). If you're using port 0, the system will choose an available port, which can be different each time you run your application.
6. Save your changes.
7. Download the `client_secret.json` file and paste it in the root directory as it is ignored in .gitignore.

Edit the file `facerecognition-google-photos.py` and make changes for the image filter from Google photos.
