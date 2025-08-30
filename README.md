# Byte meets Vibe

Protect every face around you—our AI automatically detects and blurs bystander faces in real time, keeping your world private without missing a moment.

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
1. Download [YOLOv8m-Face (ONNX) model](https://github.com/lindevs/yolov8-face) and [w600k_r50 (Arcface)](https://huggingface.co/maze/faceX/blob/e010b5098c3685fd00b22dd2aec6f37320e3d850/w600k_r50.onnx)
2. Run the calibration_gpu_rt.py with the bash commands
```
python calibrate_gpu_rt.py --video_path <vid_path> --streamer_id <id>
```
3. After calibration is complete, simply run the live_gpu_rt.py!
