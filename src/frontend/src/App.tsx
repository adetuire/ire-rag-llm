import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout.tsx';
import ChatPage from './pages/ChatPage.tsx';
import SavedAnswersPage from './pages/SavedAnswersPage.tsx';

const App = () => (
  <AppLayout>
    <Routes>
      <Route path="/" element={<ChatPage />} />
      <Route path="/chat/:sessionId" element={<ChatPage />} />
      <Route path="/saved-answers" element={<SavedAnswersPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </AppLayout>
);

export default App;
