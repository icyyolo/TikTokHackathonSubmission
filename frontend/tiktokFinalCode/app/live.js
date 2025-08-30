import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Dimensions,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { Video } from 'expo-av';
import { io } from 'socket.io-client';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';

const { width, height } = Dimensions.get('window');
const BACKEND_URL = 'https://74eed3b8b7a8.ngrok-free.app';

// Video streaming configuration
const STREAM_DELAY_SECONDS = 10;
const VIDEO_CHUNK_DURATION = 2000; // milliseconds
const VIDEO_QUALITY = 'hd'; // 'sd', 'hd', 'full-hd', '4k'

export default function VideoProcessingApp() {
  // Camera permissions and setup
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [microphonePermission, requestMicrophonePermission] = useMicrophonePermissions();
  const [facing, setFacing] = useState(CameraType.back);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);

  // Video streaming state
  const [isVideoStreaming, setIsVideoStreaming] = useState(false);
  const [processedVideoUri, setProcessedVideoUri] = useState(null);
  const [streamStats, setStreamStats] = useState({
    chunksSent: 0,
    chunksReceived: 0,
    bufferSize: 0,
    errors: 0,
    currentDelay: 0
  });

  // Video buffer for delayed playback
  const videoBuffer = useRef([]);
  const bufferPlaybackInterval = useRef(null);
  const streamingInterval = useRef(null);
  const videoCounter = useRef(0);
  const isVideoStreamingRef = useRef(false);

  // Socket connection
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [videoCount, setVideoCount] = useState(0);

  const hasPermissions = cameraPermission?.granted && microphonePermission?.granted;

  useEffect(() => {
    initializeSocket();

    return () => {
      cleanupResources();
    };
  }, []);

  useEffect(() => {
    // Camera is ready when permissions are granted
    setIsCameraReady(hasPermissions);
  }, [hasPermissions]);

  const cleanupResources = useCallback(() => {
    if (socket) {
      socket.disconnect();
    }
    if (streamingInterval.current) {
      clearInterval(streamingInterval.current);
    }
    if (bufferPlaybackInterval.current) {
      clearInterval(bufferPlaybackInterval.current);
    }
    videoBuffer.current = [];
  }, [socket]);

  const addVideoToBuffer = useCallback((videoData) => {
    const timestamp = Date.now();
    const videoChunk = {
      ...videoData,
      bufferTimestamp: timestamp,
      playTime: timestamp + (STREAM_DELAY_SECONDS * 1000)
    };

    videoBuffer.current.push(videoChunk);

    if (videoBuffer.current.length > 20) {
      videoBuffer.current = videoBuffer.current.slice(-20);
    }

    setStreamStats(prev => ({
      ...prev,
      bufferSize: videoBuffer.current.length
    }));
  }, []);

  const processDelayedVideos = useCallback(() => {
    const now = Date.now();
    const videosToPlay = [];
    const remainingVideos = [];

    videoBuffer.current.forEach(video => {
      if (video.playTime <= now) {
        videosToPlay.push(video);
      } else {
        remainingVideos.push(video);
      }
    });

    if (videosToPlay.length > 0) {
      const latestVideo = videosToPlay[videosToPlay.length - 1];
      setProcessedVideoUri(latestVideo.video_url);

      setStreamStats(prev => ({
        ...prev,
        chunksReceived: prev.chunksReceived + videosToPlay.length,
        bufferSize: remainingVideos.length,
        currentDelay: Math.round((now - latestVideo.bufferTimestamp) / 1000)
      }));
    }

    videoBuffer.current = remainingVideos;
  }, []);

  const startVideoBufferPlayback = useCallback(() => {
    if (bufferPlaybackInterval.current) {
      clearInterval(bufferPlaybackInterval.current);
    }

    bufferPlaybackInterval.current = setInterval(() => {
      if (isVideoStreamingRef.current) {
        processDelayedVideos();
      }
    }, 500);
  }, [processDelayedVideos]);

  const stopVideoBufferPlayback = useCallback(() => {
    if (bufferPlaybackInterval.current) {
      clearInterval(bufferPlaybackInterval.current);
      bufferPlaybackInterval.current = null;
    }
    videoBuffer.current = [];
  }, []);

  const initializeSocket = useCallback(() => {
    try {
      const newSocket = io(BACKEND_URL, {
        transports: ['websocket'],
        timeout: 10000,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        maxHttpBufferSize: 50 * 1024 * 1024,
      });

      newSocket.on('connect', () => {
        console.log('Connected to backend');
        setIsConnected(true);
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from backend');
        setIsConnected(false);
        stopVideoStreaming();
      });

      newSocket.on('video_stream_started', (data) => {
        console.log('Video streaming started:', data);
      });

      newSocket.on('video_stream_stopped', (data) => {
        console.log('Video streaming stopped:', data);
      });

      newSocket.on('processed_video_chunk', (data) => {
        console.log('Received processed video chunk');
        addVideoToBuffer(data);
      });

      newSocket.on('video_processed', (data) => {
        console.log('Received processed video');
        setProcessedVideoUri(data.processed_video_url);
        setVideoCount(prev => prev + 1);
        setIsProcessing(false);
      });

      newSocket.on('error', (error) => {
        console.error('Backend error:', error.message);
        Alert.alert('Processing Error', error.message);
        setIsProcessing(false);
        stopVideoStreaming();
        setStreamStats(prev => ({
          ...prev,
          errors: prev.errors + 1
        }));
      });

      setSocket(newSocket);
    } catch (error) {
      console.error('Socket connection error:', error);
      Alert.alert('Connection Error', 'Failed to connect to server');
    }
  }, [addVideoToBuffer]);

  // Record video chunk using Expo Camera
  const recordVideoChunk = useCallback(async () => {
    if (!cameraRef.current || !isCameraReady || !isVideoStreamingRef.current) {
      return null;
    }

    try {
      console.log(`Recording ${VIDEO_CHUNK_DURATION}ms video chunk...`);

      const video = await cameraRef.current.recordAsync({
        maxDuration: VIDEO_CHUNK_DURATION / 1000,
        quality: '720p',
        mute: false,
      });

      if (isVideoStreamingRef.current && video.uri) {
        await sendVideoChunk(video.uri);
      }

      return video;

    } catch (error) {
      console.warn(`Video chunk recording failed:`, error.message);
      setStreamStats(prev => ({
        ...prev,
        errors: prev.errors + 1
      }));
      return null;
    }
  }, [isCameraReady]);

  const sendVideoChunk = useCallback(async (videoUri) => {
    if (!socket || !isConnected || !videoUri) {
      return;
    }

    try {
      const fileInfo = await FileSystem.getInfoAsync(videoUri);
      if (!fileInfo.exists || fileInfo.size > 10 * 1024 * 1024) {
        console.warn('Video chunk too large or does not exist, skipping');
        return;
      }

      const videoData = await FileSystem.readAsStringAsync(videoUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      videoCounter.current += 1;

      socket.emit('video_stream_chunk', {
        video_data: videoData,
        chunk_id: videoCounter.current,
        timestamp: Date.now(),
        duration: VIDEO_CHUNK_DURATION / 1000,
        format: 'mp4'
      });

      setStreamStats(prev => ({
        ...prev,
        chunksSent: prev.chunksSent + 1
      }));

      // Clean up local file
      try {
        await FileSystem.deleteAsync(videoUri);
      } catch (deleteError) {
        console.warn('Failed to delete video chunk:', deleteError);
      }

    } catch (error) {
      console.error('Video chunk upload error:', error);
      setStreamStats(prev => ({
        ...prev,
        errors: prev.errors + 1
      }));
    }
  }, [socket, isConnected]);

  const startVideoStreaming = useCallback(async () => {
    if (!socket || !isConnected || !isCameraReady) {
      Alert.alert('Error', 'Not ready for video streaming');
      return;
    }

    if (isRecording) {
      Alert.alert('Error', 'Cannot stream while recording');
      return;
    }

    if (isVideoStreamingRef.current) {
      return;
    }

    try {
      console.log(`Starting video streaming with ${VIDEO_CHUNK_DURATION}ms chunks...`);

      setIsVideoStreaming(true);
      isVideoStreamingRef.current = true;

      setProcessedVideoUri(null);
      videoBuffer.current = [];
      videoCounter.current = 0;

      socket.emit('start_video_stream', {
        chunk_duration: VIDEO_CHUNK_DURATION / 1000,
        quality: VIDEO_QUALITY,
        delay_seconds: STREAM_DELAY_SECONDS
      });

      startVideoBufferPlayback();

      // Start streaming loop
      streamingInterval.current = setInterval(() => {
        if (isVideoStreamingRef.current) {
          recordVideoChunk();
        }
      }, VIDEO_CHUNK_DURATION + 100); // Small buffer between recordings

    } catch (error) {
      console.error('Error starting video streaming:', error);
      Alert.alert('Streaming Error', 'Failed to start video streaming');
      setIsVideoStreaming(false);
      isVideoStreamingRef.current = false;
    }
  }, [socket, isConnected, isCameraReady, isRecording, startVideoBufferPlayback, recordVideoChunk]);

  const stopVideoStreaming = useCallback(() => {
    if (!isVideoStreamingRef.current) {
      return;
    }

    console.log('Stopping video streaming...');

    isVideoStreamingRef.current = false;
    setIsVideoStreaming(false);

    if (streamingInterval.current) {
      clearInterval(streamingInterval.current);
      streamingInterval.current = null;
    }

    // Stop current recording if any
    if (cameraRef.current && isRecording) {
      try {
        cameraRef.current.stopRecording();
      } catch (error) {
        console.warn('Error stopping current recording:', error);
      }
    }

    stopVideoBufferPlayback();

    if (socket) {
      socket.emit('stop_video_stream');
    }

    videoCounter.current = 0;

    setStreamStats({
      chunksSent: 0,
      chunksReceived: 0,
      bufferSize: 0,
      errors: 0,
      currentDelay: 0
    });
  }, [socket, stopVideoBufferPlayback, isRecording]);

  const toggleVideoStreaming = useCallback(async () => {
    if (isVideoStreamingRef.current) {
      stopVideoStreaming();
    } else {
      await startVideoStreaming();
    }
  }, [startVideoStreaming, stopVideoStreaming]);

  // Regular video recording
  const startRecording = useCallback(async () => {
    if (!socket || !isConnected || !isCameraReady) {
      Alert.alert('Error', 'Not ready to record');
      return;
    }

    if (isVideoStreaming) {
      Alert.alert('Error', 'Cannot record while streaming');
      return;
    }

    try {
      setIsRecording(true);
      setIsProcessing(true);
      setProcessedVideoUri(null);

      console.log('Starting video recording...');

      const video = await cameraRef.current.recordAsync({
        maxDuration: 5,
        quality: '720p',
        mute: false,
      });

      console.log('Recording completed:', video.uri);
      setIsRecording(false);
      await sendVideoToBackend(video.uri);

    } catch (error) {
      console.error('Recording error:', error);
      setIsRecording(false);
      setIsProcessing(false);
      Alert.alert('Error', `Recording failed: ${error.message}`);
    }
  }, [socket, isConnected, isCameraReady, isVideoStreaming]);

  const sendVideoToBackend = useCallback(async (videoUri) => {
    if (!socket || !isConnected) {
      setIsProcessing(false);
      return;
    }

    try {
      console.log('Reading video file...');

      const fileInfo = await FileSystem.getInfoAsync(videoUri);
      if (!fileInfo.exists || fileInfo.size > 50 * 1024 * 1024) {
        throw new Error('Video file too large or does not exist');
      }

      const videoData = await FileSystem.readAsStringAsync(videoUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      console.log('Sending video to backend...');
      socket.emit('process_video', {
        video_data: videoData,
        video_id: Date.now(),
        format: 'mp4'
      });

      try {
        await FileSystem.deleteAsync(videoUri);
      } catch (deleteError) {
        console.warn('Failed to delete local video file:', deleteError);
      }

    } catch (error) {
      console.error('Video upload error:', error);
      setIsProcessing(false);
      Alert.alert('Upload Error', 'Failed to send video to server');
    }
  }, [socket, isConnected]);

  const stopRecording = useCallback(async () => {
    if (cameraRef.current && isRecording) {
      try {
        cameraRef.current.stopRecording();
      } catch (error) {
        console.error('Error stopping recording:', error);
      }
    }
    setIsRecording(false);
  }, [isRecording]);

  const toggleCameraFacing = useCallback(() => {
    if (isRecording) {
      stopRecording();
    }
    if (isVideoStreaming) {
      stopVideoStreaming();
    }

    setFacing(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }, [isRecording, isVideoStreaming, stopRecording, stopVideoStreaming]);

  const requestPermissions = async () => {
    if (!cameraPermission?.granted) {
      await requestCameraPermission();
    }
    if (!microphonePermission?.granted) {
      await requestMicrophonePermission();
    }
  };

  if (!hasPermissions) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Camera and microphone permissions are required</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermissions}>
          <Text style={styles.buttonText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Expo Camera App</Text>
        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: isConnected ? '#4CAF50' : '#f44336' }]} />
          <Text style={styles.statusText}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </Text>
          <View style={[styles.statusDot, { backgroundColor: isCameraReady ? '#4CAF50' : '#ffa500', marginLeft: 10 }]} />
          <Text style={styles.statusText}>Camera</Text>
        </View>
      </View>

      <View style={styles.videoContainer}>
        <View style={styles.cameraContainer}>
          <Text style={styles.sectionTitle}>Live Camera</Text>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
            mode="video"
            onCameraReady={() => setIsCameraReady(true)}
          />
          <TouchableOpacity
            style={[
              styles.flipButton,
              (isRecording || isVideoStreaming) && styles.flipButtonDisabled
            ]}
            onPress={toggleCameraFacing}
            disabled={isRecording || isVideoStreaming}
          >
            <Text style={styles.flipButtonText}>Flip</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.processedContainer}>
          <Text style={styles.sectionTitle}>
            {isVideoStreaming ? `Delayed Video Stream (-${STREAM_DELAY_SECONDS}s)` : 'Processed Video'}
          </Text>
          <View style={styles.processedVideoArea}>
            {processedVideoUri ? (
              <Video
                ref={videoRef}
                source={{ uri: processedVideoUri }}
                style={styles.processedVideo}
                resizeMode="cover"
                shouldPlay={false}
                isLooping={false}
                isMuted
              />
            ) : (
              <View style={styles.placeholderContainer}>
                <Text style={styles.placeholderText}>
                  {isVideoStreaming ? `Delayed video stream (${STREAM_DELAY_SECONDS}s behind)...` :
                    isProcessing ? 'Processing video...' :
                      'Record a video or start streaming'}
                </Text>
                {(isProcessing || isVideoStreaming) && (
                  <ActivityIndicator size="small" color="#0066cc" style={{ marginTop: 10 }} />
                )}
              </View>
            )}
          </View>
        </View>
      </View>

      {/* Streaming Controls */}
      <View style={styles.streamingContainer}>
        <Text style={styles.sectionTitle}>
          Video Streaming ({STREAM_DELAY_SECONDS}s delay, {VIDEO_CHUNK_DURATION / 1000}s chunks)
        </Text>

        <TouchableOpacity
          style={[
            styles.streamButton,
            isVideoStreaming ? styles.streamButtonActive : styles.streamButtonInactive,
            (!isConnected || !isCameraReady || isRecording) && styles.streamButtonDisabled
          ]}
          onPress={toggleVideoStreaming}
          disabled={!isConnected || !isCameraReady || isRecording}
        >
          <Text style={styles.streamButtonText}>
            {isVideoStreaming ? 'Stop Video Stream' : 'Start Video Stream'}
          </Text>
        </TouchableOpacity>

        {isVideoStreaming && (
          <View style={styles.statsContainer}>
            <Text style={styles.statsText}>
              Sent: {streamStats.chunksSent} | Buffer: {streamStats.bufferSize} | Played: {streamStats.chunksReceived}
            </Text>
            <Text style={styles.statsText}>
              Current Delay: {streamStats.currentDelay}s | Errors: {streamStats.errors}
            </Text>
          </View>
        )}
      </View>

      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording ? styles.recordButtonActive : styles.recordButtonInactive,
            (!isConnected || !isCameraReady || isProcessing || isVideoStreaming) && styles.recordButtonDisabled
          ]}
          onPress={isRecording ? stopRecording : startRecording}
          disabled={!isConnected || !isCameraReady || (isProcessing && !isRecording) || isVideoStreaming}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? 'Stop Recording' : 'Start Recording (5s)'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.infoText}>Videos processed: {videoCount}</Text>

        {isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>RECORDING</Text>
          </View>
        )}

        {isVideoStreaming && (
          <View style={styles.recordingIndicator}>
            <View style={[styles.recordingDot, { backgroundColor: '#0066cc' }]} />
            <Text style={[styles.recordingText, { color: '#0066cc' }]}>STREAMING</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
    paddingBottom: 16,
    backgroundColor: '#1a1a1a',
  },
  title: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  statusContainer: { flexDirection: 'row', alignItems: 'center' },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusText: { color: '#fff', fontSize: 12 },
  videoContainer: { flex: 1, flexDirection: 'row', padding: 8 },
  cameraContainer: { flex: 1, marginRight: 4, position: 'relative' },
  processedContainer: { flex: 1, marginLeft: 4 },
  sectionTitle: { color: '#fff', fontSize: 12, fontWeight: '600', marginBottom: 8, textAlign: 'center' },
  camera: { flex: 1, borderRadius: 8 },
  flipButton: {
    position: 'absolute', top: 8, right: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  flipButtonDisabled: { backgroundColor: 'rgba(102, 102, 102, 0.5)' },
  flipButtonText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  processedVideoArea: { flex: 1, backgroundColor: '#1a1a1a', borderRadius: 8, overflow: 'hidden' },
  processedVideo: { width: '100%', height: '100%' },
  placeholderContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  placeholderText: { color: '#666', fontSize: 12, textAlign: 'center', lineHeight: 16 },
  streamingContainer: {
    backgroundColor: '#2a2a2a',
    padding: 16,
    marginHorizontal: 8,
    marginBottom: 8,
    borderRadius: 8,
  },
  streamButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 24,
    marginBottom: 12,
    minWidth: 160,
    alignSelf: 'center',
  },
  streamButtonActive: { backgroundColor: '#ff4444' },
  streamButtonInactive: { backgroundColor: '#4444ff' },
  streamButtonDisabled: { backgroundColor: '#666' },
  streamButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', textAlign: 'center' },
  statsContainer: { backgroundColor: '#333', padding: 8, borderRadius: 6 },
  statsText: { color: '#ccc', fontSize: 12, textAlign: 'center' },
  controlsContainer: { padding: 16, alignItems: 'center', backgroundColor: '#1a1a1a' },
  recordButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 24,
    marginBottom: 12,
    minWidth: 160,
  },
  recordButtonActive: { backgroundColor: '#f44336' },
  recordButtonInactive: { backgroundColor: '#4CAF50' },
  recordButtonDisabled: { backgroundColor: '#666' },
  recordButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold', textAlign: 'center' },
  infoText: { color: '#ccc', fontSize: 12, marginBottom: 8 },
  recordingIndicator: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  recordingDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#f44336', marginRight: 6 },
  recordingText: { color: '#f44336', fontSize: 12, fontWeight: 'bold' },
  errorText: { color: '#f44336', fontSize: 14, textAlign: 'center', marginBottom: 16 },
  button: { backgroundColor: '#0066cc', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 6 },
  buttonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
});