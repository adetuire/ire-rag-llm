import { PropsWithChildren } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth.ts';

const navigation = [
  { label: 'Chat', to: '/' },
  { label: 'Saved answers', to: '/saved-answers' }
];

export const AppLayout = ({ children }: PropsWithChildren) => {
  const { user, isAuthenticating, logout } = useAuth();

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="text-lg font-semibold text-primary-600">
            IRE RAG Console
          </Link>
          <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
            <div className="flex gap-4">
              {navigation.map((item) => (
                <Link key={item.to} to={item.to} className="hover:text-primary-600">
                  {item.label}
                </Link>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              {isAuthenticating ? (
                <span>Loading account…</span>
              ) : user ? (
                <>
                  <span className="font-semibold text-slate-700">{user.name}</span>
                  <button
                    type="button"
                    onClick={() => logout().catch(console.error)}
                    className="rounded border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-600 hover:border-primary-500"
                  >
                    Sign out
                  </button>
                </>
              ) : (
                <span>Not signed in</span>
              )}
            </div>
          </nav>
        </div>
      </header>
      <main className="mx-auto flex max-w-6xl flex-1 flex-col gap-6 px-6 py-6">{children}</main>
    </div>
  );
};
