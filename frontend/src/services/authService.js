export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'https://api.amiarobot.ca';
async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`;
  const defaultHeaders = { 'Content-Type': 'application/json' };
  const mergedOptions = {
    method: 'GET',
    headers: { ...defaultHeaders, ...(options.headers || {}) },
    ...options,
  };

  const response = await fetch(url, mergedOptions);
  const isJson = response.headers.get('content-type')?.includes('application/json');
  const body = isJson ? await response.json().catch(() => null) : await response.text().catch(() => null);

  if (!response.ok) {
    const error = new Error(body?.detail || response.statusText || 'Request failed');
    error.response = response;
    error.data = body;
    throw error;
  }

  return body;
}

async function login(username, password) {
  return request('/login/', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

async function register(username, email, password) {
  return request('/register/', {
    method: 'POST',
    body: JSON.stringify({ username, email, password }),
  });
}

async function getCurrentUser(token) {
  return request('/me/', {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
}

async function requestPasswordReset(email) {
  return request('/request-password-reset/', {
    method: 'POST',
    body: JSON.stringify({ email }),
  });
}

async function resetPassword(token, newPassword) {
  return request('/reset-password/', {
    method: 'POST',
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

async function predict(data, token) {
  const headers = {
    'Content-Type': 'application/json'
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  return request('/predict/', {
    method: 'POST',
    body: JSON.stringify(data),
    headers,
  });
}

async function saveSubmission(submissionData, token = null) {
  const headers = {
    'Content-Type': 'application/json',
  };
  
  // Only add auth header if token exists
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  return request('/submissions/', {
    method: 'POST',
    body: JSON.stringify(submissionData),
    headers,
  });
}

async function getUserSubmissions(token, limit = 50) {
  const response = await request(`/submissions/?limit=${limit}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return response;
}

export const authService = { 
  login, 
  register, 
  getCurrentUser, 
  requestPasswordReset, 
  resetPassword,
  predict,
  saveSubmission,
  getUserSubmissions
};