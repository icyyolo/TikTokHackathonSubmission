# Byte meets Vibe

Protect every face around you—our AI automatically detects and blurs bystander faces, and location indicators in real time, keeping your world private without missing a moment.

## How we built it
Byte Meets Vibe has 2 components: the Frontend (React Native/Expo) and the Backend (Python Flask).

## Frontend

### Getting Started
To get the expo go server running for the frontend\
Do the following in the terminal: 
```
cd frontend/tiktokFinalCode
npm install
npx expo start --tunnel
```
On your smartphone, install **Expo Go** \
If you are using **Android**, scan using the Expo Go app\
If you are using **IOS**, scan using your phone camera\

## Backend (For manual local testing/usage)

## Face Filter
[Watch Video Here](https://youtu.be/RJbfGNuO_y8)
1. Make sure you have a Nvidia GPU, and run the scripts on Windows.
2. Download [Nvidia CUDA v12.4](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)  
3. Download [YOLOv8m-Face (ONNX) model](https://github.com/lindevs/yolov8-face) and [w600k_r50 (Arcface)](https://huggingface.co/maze/faceX/blob/e010b5098c3685fd00b22dd2aec6f37320e3d850/w600k_r50.onnx)
4. Upload a video of yourself to calibrate the model. Make sure to do the following:
Neutral => Maintain a relaxed, neutral facial expression.  
Neutral, Front-Facing => Look directly at the camera with a neutral expression.  
Head Turns (L & R) => Slowly turn your head left, then right.  
Head Tilts (Up & Down) => Slowly tilt your head up, then down.  
Head Rolls (clockwise) => Gently roll your head from side to side aka draw a big circle with your nose 
Head cross => Draw a big X with your nose
Smile (with & without) => Smile naturally, first with lips closed, then with a full smile showing teeth.  
Frown 
Eyes Closed => Briefly close your eyes.  
Raised Eyebrows => Raise your eyebrows in a surprised expression.  
Puffed Cheeks => Puff out your cheeks.
   
   Neutral => Maintain a relaxed, neutral facial expression.  
   Neutral, Front-Facing => Look directly at the camera with a neutral expression.  
   Head Turns (L & R) => Slowly turn your head left, then right.  
   Head Tilts (Up & Down) => Slowly tilt your head up, then down.  
   Head Rolls (clockwise) => Gently roll your head from side to side aka draw a big circle with your nose 
   Head cross => Draw a big X with your nose
   
   Smile (with & without) => Smile naturally, first with lips closed, then with a full smile showing teeth.  
   Frown 
   Eyes Closed => Briefly close your eyes.  
   Raised Eyebrows => Raise your eyebrows in a surprised expression.  
   Puffed Cheeks => Puff out your cheeks.
   Take-off-spectacles/Normal pose
5. Set up a virtual environment with the bash commands
```
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```
6. Run the scripts 
```
python calibrate_gpu_rt.py --video_path "your-video-path.mp4" --streamer_id "your-name" --save_debug_images #To calibrate model
```
7. After calibration is complete, simply run the live streaming script!
```
python live_gpu_rt.py
```

### Stuff we have done!
- Integrated YOLOv8m-face model for high-accuracy face detection
- Fine-tuned an ArcFace model to generate consistent, normalised 512-dimensional embeddings. The mean vector embedding is then stored in a database.
- Facial Recognition is performed using the ArcFace model by calculating the cosine similarity between a detected face's embedding and the stored embeddings of known individuals.
- Created a multi-criteria scoring system for automated quality assurance. It automatically validates samples based on Laplacian variance (sharpness/blur), brightness, contrast, and face size to filter out poor-quality images.
- Used Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes for the same face.
- Adopted advanced GPU acceleration with TensorRT for maximum performance by transitioning our pipeline to TensorRT 10.13, using FP16 precision to accelerate inference. 
- Shifted to a single-threaded processing model to avoid CUDA context conflicts, while still using thread-safe queues for ingesting frames from the camera.
- Stabilised the filtered live stream with a 5-second buffer delay, which absorbs processing time variations and presents the user with a consistently smooth video stream rather than a choppy, lagging one.

### Challenges faced
- While finding ways to reduce lag, we have tested many techniques such as frame skipping, reducing the frequency of object recognition, multi-threading, including a buffer delay, using a whole different model. Nevertheless, we were almost always frequently led to a whole other nest of problems such as bounding box delays, jittering, frame rate volatility and even the infamous dependency hell :(
- 

## Text Filter
1. The application can process both a live webcam feed and a pre-recorded MP4 video file.
2. To Use a Webcam:
   Run the script with no arguments
```
python text_filter.py
```
3. To Process a Video File:
   Provide the path to the video file as a command-line argument.
```
python text_filter.py path/to/your/video.mp4
```
### Features
- Real-time text detection with PaddleOCR.
- Automatic Gaussian blurring of detected regions.
- Pre-processing (sharpening) to improve OCR accuracy on blurry text.
- Dual configuration modes: QUALITY (better detection) and FAST (better FPS).
- Works with webcam or video file input.
- Hybrid "Detect-and-Track" Architecture: Every 12 frames, the PaddleOCR model performs a high-quality scan of the scene to find all text. For the frames in between, a set of Channel and Spatial Reliability Tracking (CSRT) trackers take over. These trackers smoothly follow the objects identified, and ensure the blur locks on to moving text.

### Demo
<img width="2559" height="1439" alt="no filter" src="https://github.com/user-attachments/assets/0da2efb7-d760-4ebd-b101-1902d18aa19d" />   

Location of video revealed: Khatib Vale 

<img width="699" height="677" alt="no filter image search" src="https://github.com/user-attachments/assets/99c2ae0e-2b67-4691-9531-750eb38481fa" /> 

Google image search was able to identify the location accurately 


<img width="2559" height="1439" alt="with filter" src="https://github.com/user-attachments/assets/5b939a25-70cd-4114-91e5-9609fe239e00" /> 

Location of video blurred 

<img width="718" height="695" alt="with filter image search" src="https://github.com/user-attachments/assets/33356d76-e8d0-4d48-a9a1-eed41940e789" /> 

Google image search unable to identify the location accurately
