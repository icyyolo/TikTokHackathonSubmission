import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  SafeAreaView,
  StatusBar,
  Image,
  ActivityIndicator,
  Dimensions,
  Platform,
  Linking,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { Video, VideoView } from 'expo-video';
import * as FileSystem from 'expo-file-system';

const { width, height } = Dimensions.get('window');

export default function MediaGalleryApp() {
  const [media, setMedia] = useState([]);
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [hasPermission, setHasPermission] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    (async () => {
      // Request permissions
      const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
      const { status: mediaStatus } = await MediaLibrary.requestPermissionsAsync();
      
      setHasPermission(cameraStatus === 'granted' && mediaStatus === 'granted');

      if (cameraStatus !== 'granted' || mediaStatus !== 'granted') {
        Alert.alert(
          'Permission required', 
          'Camera and photo library permissions are required to use this app',
          [
            {
              text: 'Open Settings',
              onPress: () => {
                if (Platform.OS === 'ios') {
                  Linking.openURL('app-settings:');
                } else {
                  Linking.openSettings();
                }
              }
            },
            { text: 'Cancel', style: 'cancel' }
          ]
        );
      }
    })();
  }, []);

  const loadMedia = async () => {
    setLoading(true);
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Photo library access is required to load media');
        setLoading(false);
        return;
      }

      const { assets } = await MediaLibrary.getAssetsAsync({
        mediaType: [MediaLibrary.MediaType.photo, MediaLibrary.MediaType.video],
        sortBy: [[MediaLibrary.SortBy.creationTime, false]],
        first: 50,
      });

      // Process assets to get proper URIs and metadata
      const processedAssets = await Promise.all(
        assets.map(async (asset) => {
          try {
            const assetInfo = await MediaLibrary.getAssetInfoAsync(asset.id);
            
            // For videos, we need to use the localUri property
            const uri = assetInfo.localUri || asset.uri;
            
            return {
              ...asset,
              uri: uri,
              mediaType: asset.mediaType === MediaLibrary.MediaType.video ? 'video' : 'photo',
              duration: assetInfo.duration,
              filename: asset.filename,
              width: asset.width,
              height: asset.height
            };
          } catch (error) {
            console.error('Error processing asset:', error);
            return {
              ...asset,
              uri: asset.uri,
              mediaType: asset.mediaType === MediaLibrary.MediaType.video ? 'video' : 'photo',
              filename: asset.filename
            };
          }
        })
      );

      setMedia(processedAssets);
    } catch (error) {
      console.error('Error loading media:', error);
      Alert.alert('Error', `Failed to load media: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const pickMedia = async () => {
    try {
      let result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: false,
        quality: 1,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const newMedia = result.assets[0];
        setMedia(prevMedia => [newMedia, ...prevMedia]);
      }
    } catch (error) {
      console.error('Error picking media:', error);
      Alert.alert('Error', 'Failed to pick media');
    }
  };

  const takePhoto = async () => {
    try {
      let result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const newMedia = result.assets[0];
        setMedia(prevMedia => [newMedia, ...prevMedia]);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const recordVideo = async () => {
    try {
      let result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const newMedia = result.assets[0];
        setMedia(prevMedia => [newMedia, ...prevMedia]);
      }
    } catch (error) {
      console.error('Error recording video:', error);
      Alert.alert('Error', 'Failed to record video');
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const getMediaType = (item) => {
    if (item.mediaType) return item.mediaType;
    if (item.type) return item.type;
    
    if (item.uri) {
      const uri = item.uri.toLowerCase();
      if (uri.includes('.mp4') || uri.includes('.mov') || uri.includes('.avi')) {
        return 'video';
      }
      return 'photo';
    }
    
    return 'photo';
  };

  if (hasPermission === null) {
    return (
      <View style={styles.centerContainer}>
        <Text>Requesting permissions...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.centerContainer}>
        <Text>No access to camera or photo library</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />

      {selectedMedia ? (
        <View style={styles.fullScreenContainer}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => {
              if (videoRef.current) {
                videoRef.current.pause();
              }
              setSelectedMedia(null);
            }}
          >
            <Text style={styles.closeButtonText}>✕</Text>
          </TouchableOpacity>

          <View style={styles.mediaContainer}>
            {getMediaType(selectedMedia) === 'video' ? (
              <VideoView
                ref={videoRef}
                style={styles.fullMedia}
                source={{ uri: selectedMedia.uri }}
                allowsFullscreen
                allowsPictureInPicture
                nativeControls
              />
            ) : (
              <Image
                source={{ uri: selectedMedia.uri }}
                style={styles.fullMedia}
                resizeMode="contain"
              />
            )}
          </View>

          <View style={styles.infoContainer}>
            {getMediaType(selectedMedia) === 'video' && selectedMedia.duration && (
              <Text style={styles.mediaInfo}>
                Video • {formatDuration(selectedMedia.duration)}
              </Text>
            )}
            {getMediaType(selectedMedia) === 'photo' && selectedMedia.width && selectedMedia.height && (
              <Text style={styles.mediaInfo}>
                Photo • {selectedMedia.width} × {selectedMedia.height}
              </Text>
            )}
            {selectedMedia.filename && (
              <Text style={styles.fileName} numberOfLines={1}>
                {selectedMedia.filename}
              </Text>
            )}
          </View>
        </View>
      ) : (
        <>
          <View style={styles.header}>
            <Text style={styles.title}>Media Gallery</Text>
          </View>

          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.button} onPress={pickMedia}>
              <Text style={styles.buttonText}>Pick Media</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.button, styles.cameraButton]} onPress={takePhoto}>
              <Text style={styles.buttonText}>📸 Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.button, styles.videoButton]} onPress={recordVideo}>
              <Text style={styles.buttonText}>📹 Video</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.refreshContainer}>
            <TouchableOpacity style={styles.refreshButton} onPress={loadMedia}>
              <Text style={styles.refreshButtonText}>Load Gallery</Text>
            </TouchableOpacity>
          </View>

          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>Loading media...</Text>
            </View>
          ) : media.length === 0 ? (
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>Your gallery is empty</Text>
              <Text style={styles.emptySubtext}>Start by adding photos or videos</Text>
            </View>
          ) : (
            <ScrollView style={styles.galleryContainer}>
              <View style={styles.mediaGrid}>
                {media.map((item, index) => (
                  <TouchableOpacity
                    key={`${item.id || index}`}
                    style={styles.mediaItem}
                    onPress={() => setSelectedMedia(item)}
                  >
                    {getMediaType(item) === 'video' ? (
                      <View style={styles.videoThumbnailContainer}>
                        <Image
                          source={{ uri: item.uri }}
                          style={styles.thumbnail}
                          resizeMode="cover"
                        />
                        <View style={styles.videoOverlay}>
                          <Text style={styles.playIcon}>▶</Text>
                          {item.duration && (
                            <Text style={styles.durationText}>
                              {formatDuration(item.duration)}
                            </Text>
                          )}
                        </View>
                      </View>
                    ) : (
                      <Image
                        source={{ uri: item.uri }}
                        style={styles.thumbnail}
                        resizeMode="cover"
                      />
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>
          )}
        </>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    padding: 15,
    justifyContent: 'space-around',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderRadius: 8,
    flex: 1,
    marginHorizontal: 3,
  },
  cameraButton: {
    backgroundColor: '#34C759',
  },
  videoButton: {
    backgroundColor: '#FF3B30',
  },
  buttonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  refreshContainer: {
    alignItems: 'center',
    padding: 10,
  },
  refreshButton: {
    backgroundColor: '#FF9500',
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  refreshButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
  },
  galleryContainer: {
    flex: 1,
  },
  mediaGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 2,
  },
  mediaItem: {
    width: '33.33%',
    aspectRatio: 1,
    padding: 2,
  },
  thumbnail: {
    width: '100%',
    height: '100%',
    borderRadius: 8,
  },
  videoThumbnailContainer: {
    width: '100%',
    height: '100%',
    borderRadius: 8,
    overflow: 'hidden',
  },
  videoOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  playIcon: {
    color: 'white',
    fontSize: 24,
    marginBottom: 5,
  },
  durationText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  fullScreenContainer: {
    flex: 1,
    backgroundColor: 'black',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
  },
  closeButtonText: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  mediaContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullMedia: {
    width: '100%',
    height: '100%',
  },
  infoContainer: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    padding: 15,
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  mediaInfo: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 5,
  },
  fileName: {
    color: 'white',
    fontSize: 14,
    textAlign: 'center',
    opacity: 0.8,
  },
});