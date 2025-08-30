import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Dimensions,
  ActivityIndicator,
  Image
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { io } from 'socket.io-client';

const { width, height } = Dimensions.get('window');

// Configure your backend server URL
const BACKEND_URL = 'https://42dad4ac025a.ngrok-free.app'; // Replace with your actual backend IP

export default function VideoProcessingApp() {
  // Camera and permissions
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraType, setCameraType] = useState('back');
  const cameraRef = useRef(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);

  // Socket connection
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  // Processed video display
  const [processedFrame, setProcessedFrame] = useState(null);
  const [frameCount, setFrameCount] = useState(0);

  // Recording interval and capture state
  const recordingInterval = useRef(null);
  const isCapturing = useRef(false);
  const captureQueue = useRef([]);
  
  // Camera ready timeout ref
  const cameraReadyTimeout = useRef(null);

  useEffect(() => {
    initializeSocket();
    
    return () => {
      if (socket) {
        socket.disconnect();
      }
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
      if (cameraReadyTimeout.current) {
        clearTimeout(cameraReadyTimeout.current);
      }
    };
  }, []);

  const initializeSocket = () => {
    try {
      const newSocket = io(BACKEND_URL, {
        transports: ['websocket'],
        timeout: 5000,
      });

      newSocket.on('connect', () => {
        console.log('Connected to backend');
        setIsConnected(true);
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from backend');
        setIsConnected(false);
      });

      newSocket.on('connection_response', (data) => {
        console.log('Backend response:', data.message);
      });

      newSocket.on('processed_video_chunk', (data) => {
        console.log(`Received processed chunk ${data.chunk_id}`);
        setProcessedFrame(data.processed_frame);
      });

      newSocket.on('processing_started', (data) => {
        console.log('Processing started on backend');
        setIsProcessing(true);
      });

      newSocket.on('processing_stopped', (data) => {
        console.log('Processing stopped on backend');
        setIsProcessing(false);
      });

      newSocket.on('error', (error) => {
        console.error('Backend error:', error.message);
        Alert.alert('Processing Error', error.message);
      });

      setSocket(newSocket);
    } catch (error) {
      console.error('Socket connection error:', error);
      Alert.alert('Connection Error', 'Failed to connect to backend server');
    }
  };

  // Handle camera ready state with timeout fallback
  const handleCameraReady = useCallback(() => {
    console.log('Camera is ready');
    setIsCameraReady(true);
    
    // Clear any existing timeout
    if (cameraReadyTimeout.current) {
      clearTimeout(cameraReadyTimeout.current);
      cameraReadyTimeout.current = null;
    }
  }, []);

  // Set up camera ready timeout when camera type changes
  useEffect(() => {
    // Clear any existing timeout
    if (cameraReadyTimeout.current) {
      clearTimeout(cameraReadyTimeout.current);
    }

    // Set a timeout to force camera ready state if callback doesn't fire
    cameraReadyTimeout.current = setTimeout(() => {
      console.log('Camera ready timeout - forcing ready state');
      setIsCameraReady(true);
    }, 2000); // 2 second timeout

    // Cleanup function
    return () => {
      if (cameraReadyTimeout.current) {
        clearTimeout(cameraReadyTimeout.current);
        cameraReadyTimeout.current = null;
      }
    };
  }, [cameraType]);

  // Queue-based frame capture to prevent concurrent captures
  const captureAndSendFrame = useCallback(async () => {
    try {
      // Skip if already capturing or camera not ready
      if (isCapturing.current || !cameraRef.current || !socket || !isConnected || !isCameraReady) {
        return;
      }

      isCapturing.current = true;

      // Capture photo from camera using the new API
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5, // Reduced quality for better performance
        base64: true,
        skipProcessing: true,
      });

      if (photo && photo.base64) {
        // Send frame to backend
        const frameData = {
          frame: `data:image/jpeg;base64,${photo.base64}`,
          chunk_id: Date.now(), // Use timestamp as unique ID
          timestamp: Date.now(),
        };

        socket.emit('video_chunk', frameData);
        setFrameCount(prev => prev + 1);
        console.log(`Sent frame ${frameCount}`);
      }
    } catch (error) {
      console.error('Error capturing frame:', error);
      // Don't show alert for every error, just log it
    } finally {
      isCapturing.current = false;
    }
  }, [socket, isConnected, isCameraReady, frameCount]);

  const startRecording = async () => {
    try {
      if (!socket || !isConnected) {
        Alert.alert('Error', 'Not connected to backend server');
        return;
      }

      if (!isCameraReady) {
        Alert.alert('Error', 'Camera is not ready yet');
        return;
      }

      setIsRecording(true);
      setFrameCount(0);
      
      // Notify backend to start processing
      socket.emit('start_processing');

      // Start capturing frames at a slower interval to prevent overwhelming the camera
      // Increased interval from 200ms to 500ms (2 FPS) for better stability
      recordingInterval.current = setInterval(() => {
        captureAndSendFrame();
      }, 100);

    } catch (error) {
      console.error('Error starting recording:', error);
      Alert.alert('Error', 'Failed to start recording');
    }
  };

  const stopRecording = useCallback(() => {
    try {
      setIsRecording(false);
      
      // Clear the recording interval
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
        recordingInterval.current = null;
      }

      // Wait for any ongoing capture to complete
      setTimeout(() => {
        isCapturing.current = false;
      }, 100);

      // Notify backend to stop processing
      if (socket && isConnected) {
        socket.emit('stop_processing');
      }

      console.log('Recording stopped');
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  }, [socket, isConnected]);

  const toggleCamera = useCallback(() => {
    // Stop recording before changing camera
    if (isRecording) {
      stopRecording();
    }
    
    // Clear any existing camera ready timeout
    if (cameraReadyTimeout.current) {
      clearTimeout(cameraReadyTimeout.current);
      cameraReadyTimeout.current = null;
    }
    
    // Reset camera ready state temporarily
    setIsCameraReady(false);
    
    // Change camera type (this will trigger the useEffect that sets up the timeout)
    setCameraType(current => (current === 'back' ? 'front' : 'back'));
    
    // Provide immediate feedback that camera is switching
    console.log('Switching camera...');
  }, [isRecording, stopRecording]);

  // Clean up when component unmounts
  useEffect(() => {
    return () => {
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
      if (cameraReadyTimeout.current) {
        clearTimeout(cameraReadyTimeout.current);
      }
      isCapturing.current = false;
    };
  }, []);

  if (!permission) {
    // Camera permissions are still loading
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0066cc" />
        <Text style={styles.loadingText}>Loading camera...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Live Video Processing</Text>
        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: isConnected ? '#4CAF50' : '#f44336' }]} />
          <Text style={styles.statusText}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </Text>
          {!isCameraReady && (
            <Text style={[styles.statusText, { marginLeft: 10, color: '#ffa500' }]}>
              Camera Loading...
            </Text>
          )}
        </View>
      </View>

      {/* Camera and Processed Video */}
      <View style={styles.videoContainer}>
        {/* Live Camera Feed */}
        <View style={styles.cameraContainer}>
          <Text style={styles.sectionTitle}>Live Camera</Text>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={cameraType}
            onCameraReady={handleCameraReady}
          />
          <TouchableOpacity 
            style={[
              styles.flipButton,
              isRecording && styles.flipButtonDisabled
            ]} 
            onPress={toggleCamera}
            disabled={isRecording}
          >
            <Text style={styles.flipButtonText}>Flip</Text>
          </TouchableOpacity>
        </View>

        {/* Processed Video Display */}
        <View style={styles.processedContainer}>
          <Text style={styles.sectionTitle}>Processed Video</Text>
          <View style={styles.processedVideoArea}>
            {processedFrame ? (
              <Image source={{ uri: processedFrame }} style={styles.processedVideo} />
            ) : (
              <View style={styles.placeholderContainer}>
                <Text style={styles.placeholderText}>
                  {isProcessing ? 'Processing...' : 'Start recording to see processed video'}
                </Text>
                {isProcessing && <ActivityIndicator size="small" color="#0066cc" />}
              </View>
            )}
          </View>
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording ? styles.recordButtonActive : styles.recordButtonInactive,
            (!isConnected || !isCameraReady) && styles.recordButtonDisabled
          ]}
          onPress={isRecording ? stopRecording : startRecording}
          disabled={!isConnected || !isCameraReady}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.infoText}>
          Frames processed: {frameCount}
        </Text>
        
        {isRecording && (
          <Text style={styles.recordingIndicator}>● RECORDING</Text>
        )}

        {!isCameraReady && (
          <Text style={styles.warningText}>
            {cameraType === 'back' ? 'Initializing back camera...' : 'Initializing front camera...'}
          </Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 20,
    backgroundColor: '#1a1a1a',
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
  },
  videoContainer: {
    flex: 1,
    flexDirection: 'row',
    padding: 10,
  },
  cameraContainer: {
    flex: 1,
    marginRight: 5,
  },
  processedContainer: {
    flex: 1,
    marginLeft: 5,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  camera: {
    flex: 1,
    borderRadius: 10,
  },
  flipButton: {
    position: 'absolute',
    top: 40,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  flipButtonDisabled: {
    backgroundColor: 'rgba(102, 102, 102, 0.5)',
  },
  flipButtonText: {
    color: '#fff',
    fontSize: 12,
  },
  processedVideoArea: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    overflow: 'hidden',
  },
  processedVideo: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  placeholderText: {
    color: '#666',
    textAlign: 'center',
    marginBottom: 10,
    fontSize: 12,
  },
  controlsContainer: {
    padding: 20,
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
  },
  recordButton: {
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 25,
    marginBottom: 15,
  },
  recordButtonActive: {
    backgroundColor: '#f44336',
  },
  recordButtonInactive: {
    backgroundColor: '#4CAF50',
  },
  recordButtonDisabled: {
    backgroundColor: '#666',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  infoText: {
    color: '#ccc',
    fontSize: 14,
    marginBottom: 10,
  },
  recordingIndicator: {
    color: '#f44336',
    fontSize: 12,
    fontWeight: 'bold',
  },
  warningText: {
    color: '#ffa500',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 10,
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
  },
  errorText: {
    color: '#f44336',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#0066cc',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});