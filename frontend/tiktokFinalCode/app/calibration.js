// app/calibration.js
import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
  Dimensions,
  ActivityIndicator,
  Modal,
  Platform,
  Linking,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker'; // Import ImagePicker for camera
// import axios from 'axios';
import * as FileSystem from 'expo-file-system';
// import { BACKEND_URL } from '../constants/config';
import { uploadVideo } from '@/constants/calibrationAPI';

const { width, height } = Dimensions.get('window');

const POSES = [
  { id: 'neutral', title: 'Neutral', instruction: 'Maintain a relaxed, neutral facial expression' },
  { id: 'front-facing', title: 'Neutral, Front-Facing', instruction: 'Look directly at the camera with a neutral expression' },
  { id: 'head-turns', title: 'Head Turns (L & R)', instruction: 'Slowly turn your head left, then right' },
  { id: 'head-tilts', title: 'Head Tilts (Up & Down)', instruction: 'Slowly tilt your head up, then down' },
  { id: 'head-rolls', title: 'Head Rolls', instruction: 'Gently roll your head in a circle (draw a big circle with your nose)' },
  { id: 'head-cross', title: 'Head Cross', instruction: 'Draw a big X with your nose' },
  { id: 'smile-closed', title: 'Smile (Lips Closed)', instruction: 'Smile naturally with lips closed' },
  { id: 'smile-open', title: 'Smile (Teeth)', instruction: 'Smile showing teeth' },
  { id: 'frown', title: 'Frown', instruction: 'Make a frowning expression' },
  { id: 'eyes-closed', title: 'Eyes Closed', instruction: 'Briefly close your eyes' },
  { id: 'eyebrows', title: 'Raised Eyebrows', instruction: 'Raise your eyebrows in a surprised expression' },
  { id: 'cheeks', title: 'Puffed Cheeks', instruction: 'Puff out your cheeks' },
  { id: 'glasses-off', title: 'Take-off Spectacles', instruction: 'Remove glasses if wearing them' },
  { id: 'normal', title: 'Normal Pose', instruction: 'Return to your normal pose' }
];

export default function CalibrationScreen() {
  const [hasPermission, setHasPermission] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [currentPoseIndex, setCurrentPoseIndex] = useState(0);
  const [completedPoses, setCompletedPoses] = useState([]);
  const [showInstructions, setShowInstructions] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [recordingError, setRecordingError] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      if (status !== 'granted') {
        Alert.alert(
          'Permission required',
          'Camera permission is needed for face calibration',
          [
            { text: 'Open Settings', onPress: () => Linking.openSettings() },
            { text: 'Cancel', style: 'cancel' }
          ]
        );
      }
    })();
  }, []);

  const startRecording = async () => {
    if (isRecording) return;

    try {
      setIsRecording(true);
      setRecordingError(null);

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        quality: 0.5,
      });

      if (!result.canceled && result.assets?.length > 0) {
        await processVideo(result.assets[0].uri);
      } else {
        setRecordingError("Recording was cancelled or failed.");
      }
    } catch (error) {
      setRecordingError(`Recording failed: ${error.message}`);
      Alert.alert('Error', `Failed to record video: ${error.message}`);
    } finally {
      setIsRecording(false);
    }
  };

  const processVideo = async (videoUri) => {
    setUploading(true);
    setRecordingError(null);

    try {
      const data = await uploadVideo(videoUri, POSES[currentPoseIndex].id);
      if (data.success) {
        setCompletedPoses(prev => [...prev, POSES[currentPoseIndex].id]);
        if (currentPoseIndex < POSES.length - 1) {
          setCurrentPoseIndex(prev => prev + 1);
        } else {
          Alert.alert('Success', 'All calibration poses completed!');
        }
      } else {
        throw new Error(data.error || 'Processing failed');
      }
    } catch (error) {
      const message = error.response?.data?.error || error.message || 'Upload failed';
      setRecordingError(`Upload failed: ${message}`);
      Alert.alert('Upload Failed', message);
    } finally {
      setUploading(false);
    }
  };

  const nextPose = () => {
    if (currentPoseIndex < POSES.length - 1) setCurrentPoseIndex(i => i + 1);
  };

  const prevPose = () => {
    if (currentPoseIndex > 0) setCurrentPoseIndex(i => i - 1);
  };

  const resetCalibration = () => {
    setCurrentPoseIndex(0);
    setCompletedPoses([]);
    setRecordingError(null);
  };

  if (hasPermission === null) {
    return <View style={styles.container}><Text style={styles.permissionText}>Requesting camera permission...</Text></View>;
  }

  if (hasPermission === false) {
    return <View style={styles.container}><Text style={styles.permissionText}>No access to camera. Please enable in settings.</Text></View>;
  }

  const currentPose = POSES[currentPoseIndex];
  const isCompleted = completedPoses.includes(currentPose.id);

  return (
    <View style={styles.container}>
      {/* Instructions Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={showInstructions}
        onRequestClose={() => setShowInstructions(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Calibration Instructions</Text>
            <ScrollView style={styles.instructionsScroll}>
              {POSES.map((pose, index) => (
                <View key={pose.id} style={styles.instructionItem}>
                  <Text style={[
                    styles.instructionText,
                    completedPoses.includes(pose.id) && styles.completedText
                  ]}>
                    {index + 1}. {pose.title}: {pose.instruction}
                  </Text>
                </View>
              ))}
            </ScrollView>
            <TouchableOpacity
              style={styles.modalButton}
              onPress={() => setShowInstructions(false)}
            >
              <Text style={styles.modalButtonText}>Start Calibration</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Face Calibration</Text>
        <Text style={styles.subtitle}>Pose: {currentPose.title}</Text>
      </View>

      {/* Camera Placeholder / Instructions */}
      <View style={styles.cameraPlaceholder}>
        <Text style={styles.cameraPlaceholderText}>
          {isRecording ? "Recording..." : "Camera will open when you tap 'Start Recording'"}
        </Text>
        {isRecording && <ActivityIndicator size="large" color="#FF3B30" />}
      </View>

      {/* Pose Instruction */}
      <View style={styles.instructionContainer}>
        <Text style={styles.instructionText}>{currentPose.instruction}</Text>
        {isCompleted && (
          <Text style={styles.completedBadge}>✓ Completed</Text>
        )}
        {recordingError && (
          <Text style={styles.errorText}>{recordingError}</Text>
        )}
      </View>

      {/* Progress */}
      <View style={styles.progressContainer}>
        <Text style={styles.progressText}>
          {currentPoseIndex + 1} of {POSES.length}
        </Text>
        <View style={styles.progressBar}>
          <View 
            style={[
              styles.progressFill, 
              { width: `${((currentPoseIndex + 1) / POSES.length) * 100}%` }
            ]} 
          />
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity 
          style={[styles.controlButton, styles.navButton]}
          onPress={prevPose}
          disabled={currentPoseIndex === 0}
        >
          <Text style={styles.controlText}>Previous</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.controlButton, 
            styles.recordButton,
            isRecording && styles.recordingButton
          ]}
          onPress={startRecording}
          disabled={isRecording || uploading}
        >
          {uploading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.controlText}>
              {isRecording ? 'Recording...' : 'Start Recording'}
            </Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.controlButton, styles.navButton]}
          onPress={nextPose}
          disabled={currentPoseIndex === POSES.length - 1}
        >
          <Text style={styles.controlText}>Next</Text>
        </TouchableOpacity>
      </View>

      {/* Bottom Controls */}
      <View style={styles.bottomControls}>
        <TouchableOpacity 
          style={styles.secondaryButton}
          onPress={resetCalibration}
        >
          <Text style={styles.secondaryText}>Reset</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.secondaryButton}
          onPress={() => setShowInstructions(true)}
        >
          <Text style={styles.secondaryText}>Instructions</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 50,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    width: '90%',
    maxHeight: '80%',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  instructionsScroll: {
    maxHeight: height * 0.6,
  },
  instructionItem: {
    paddingVertical: 5,
  },
  instructionText: {
    fontSize: 16,
    marginBottom: 5,
  },
  completedText: {
    textDecorationLine: 'line-through',
    color: 'gray',
  },
  modalButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 15,
  },
  modalButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  header: {
    padding: 15,
    alignItems: 'center',
  },
  title: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  subtitle: {
    color: 'white',
    fontSize: 18,
    marginTop: 5,
  },
  cameraPlaceholder: {
    flex: 1,
    margin: 10,
    borderRadius: 15,
    backgroundColor: '#333',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#555',
  },
  cameraPlaceholderText: {
    color: '#aaa',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 10,
  },
  instructionContainer: {
    padding: 20,
    alignItems: 'center',
  },
  instructionText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 10,
  },
  completedBadge: {
    color: '#4CD964',
    fontWeight: 'bold',
    fontSize: 16,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 5,
  },
  progressContainer: {
    marginHorizontal: 20,
    marginBottom: 20,
  },
  progressText: {
    color: 'white',
    textAlign: 'center',
    marginBottom: 5,
  },
  progressBar: {
    height: 10,
    backgroundColor: '#333',
    borderRadius: 5,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 5,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
    marginHorizontal: 10,
  },
  controlButton: {
    padding: 15,
    borderRadius: 30,
    minWidth: 120,
    alignItems: 'center',
  },
  navButton: {
    backgroundColor: '#333',
  },
  recordButton: {
    backgroundColor: '#007AFF',
  },
  recordingButton: {
    backgroundColor: '#FF3B30',
  },
  controlText: {
    color: 'white',
    fontWeight: 'bold',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  secondaryButton: {
    padding: 10,
    borderWidth: 1,
    borderColor: '#007AFF',
    borderRadius: 20,
  },
  secondaryText: {
    color: '#007AFF',
  },
});