import { create } from 'zustand';

interface User {
  id: string;
  name: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticating: boolean;
  error?: string;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  setIsAuthenticating: (value: boolean) => void;
  setError: (error?: string) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticating: false,
  error: undefined,
  setUser: (user) => set({ user }),
  setToken: (token) => set({ token }),
  setIsAuthenticating: (isAuthenticating) => set({ isAuthenticating }),
  setError: (error) => set({ error })
}));
