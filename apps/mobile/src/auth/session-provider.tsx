import * as SecureStore from 'expo-secure-store';
import { PropsWithChildren, createContext, useEffect, useMemo, useState } from 'react';
import { Platform } from 'react-native';
import { api } from '../api/client';

const SESSION_KEY = 'signtalk-session';

type User = { id: string; email: string; name: string };
type Auth = { user?: User; loading: boolean; signOut: () => Promise<void> };

const readSession = async () => {
  if (Platform.OS === 'web') return globalThis.localStorage?.getItem(SESSION_KEY) ?? null;
  return SecureStore.getItemAsync(SESSION_KEY);
};

const deleteSession = async () => {
  if (Platform.OS === 'web') {
    globalThis.localStorage?.removeItem(SESSION_KEY);
    return;
  }
  await SecureStore.deleteItemAsync(SESSION_KEY);
};

export const AuthContext = createContext<Auth>({ loading: true, signOut: async () => {} });

export function SessionProvider({ children }: PropsWithChildren) {
  const [user, setUser] = useState<User>();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void readSession()
      .then((session) => api.get('/profile', { headers: { Authorization: `Bearer ${session ?? ''}` } }))
      .then((response) => setUser(response.data.user))
      .catch(() => setUser(undefined))
      .finally(() => setLoading(false));
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      signOut: async () => {
        await deleteSession();
        setUser(undefined);
      },
    }),
    [user, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
