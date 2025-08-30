import axios from 'axios';
import { BACKEND_URL } from '@/constants/config';

export const uploadVideo = async (videoUri, poseId) => {
    const filename = `${poseId}_${Date.now()}.mov`;
    const formData = new FormData();
    formData.append('video', {
        uri: videoUri,
        type: 'video/quicktime',
        name: filename,
    });
    //   formData.append('pose', poseId);
    //   formData.append('')

    // Append form data fields
    formData.append('streamer_id', "test");
    formData.append('name', "test");
    formData.append('pose_category', poseId);
    formData.append('replace', true);

    const response = await axios.post(`${BACKEND_URL}/calibrate`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
    });

    return response.data;
};