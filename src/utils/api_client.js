// src/utils/api_client.js

import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 10000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let errorMessage = 'An unexpected error occurred.';

    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error:', error.response.status, error.response.data);
      errorMessage = `API Error: ${error.response.status} - ${error.response.data.message || 'Server error'}`; // Added message from response data
    } else if (error.request) {
      // The request was made but no response was received
      // `error.request` is an instance of XMLHttpRequest in the browser and an instance of
      // http.ClientRequest in node.js
      console.error('No response received from the server.');
      errorMessage = 'No response received from the server. Please check your network connection.';
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Error setting up the request:', error.message);
      errorMessage = `Error setting up the request: ${error.message}`; //Kept the original error message.
    }

    // Optionally, you could log the error to a service like Sentry
    // Sentry.captureException(error);

    return Promise.reject(new Error(errorMessage));
  }
);

export default apiClient;
