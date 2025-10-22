import apiClient from './client.ts';

export interface Credentials {
  username: string;
  password: string;
}

export interface AuthResponse {
  token: string;
  user: {
    id: string;
    name: string;
  };
}

export const login = async (credentials: Credentials): Promise<AuthResponse> => {
  const { data } = await apiClient.post<AuthResponse>('/auth/login', credentials);
  return data;
};

export const logout = async (): Promise<void> => {
  await apiClient.post('/auth/logout');
};

export const fetchCurrentUser = async (): Promise<AuthResponse['user']> => {
  const { data } = await apiClient.get<AuthResponse['user']>('/auth/me');
  return data;
};
