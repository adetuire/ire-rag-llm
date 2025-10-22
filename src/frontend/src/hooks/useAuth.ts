import { useCallback, useEffect } from 'react';
import { fetchCurrentUser, login, logout, type Credentials } from '../api/auth.ts';
import { useAuthStore } from '../store/authStore.ts';

export const useAuth = () => {
  const { user, token, isAuthenticating, error, setUser, setToken, setIsAuthenticating, setError } =
    useAuthStore();

  useEffect(() => {
    setIsAuthenticating(true);
    fetchCurrentUser()
      .then((currentUser) => setUser(currentUser))
      .catch((err: Error) => setError(err.message))
      .finally(() => setIsAuthenticating(false));
  }, [setUser, setIsAuthenticating, setError]);

  const handleLogin = useCallback(
    async (credentials: Credentials) => {
      setIsAuthenticating(true);
      setError(undefined);

      try {
        const response = await login(credentials);
        setToken(response.token);
        setUser(response.user);
        return response.user;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Authentication failed';
        setError(message);
        throw err;
      } finally {
        setIsAuthenticating(false);
      }
    },
    [setToken, setUser, setIsAuthenticating, setError]
  );

  const handleLogout = useCallback(async () => {
    setIsAuthenticating(true);
    setError(undefined);

    try {
      await logout();
      setToken(null);
      setUser(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to logout';
      setError(message);
      throw err;
    } finally {
      setIsAuthenticating(false);
    }
  }, [setToken, setUser, setIsAuthenticating, setError]);

  return {
    user,
    token,
    isAuthenticating,
    error,
    login: handleLogin,
    logout: handleLogout
  };
};
