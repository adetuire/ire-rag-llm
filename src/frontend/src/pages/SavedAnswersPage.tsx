import SavedAnswerList from '../components/SavedAnswerList.tsx';
import { useChatSession } from '../hooks/useChatSession.ts';

export const SavedAnswersPage = () => {
  const { savedAnswers, deleteAnswer, updateAnswer } = useChatSession();

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <SavedAnswerList
        answers={savedAnswers}
        onDelete={(answer) => deleteAnswer(answer.id)}
        onUpdate={(answer, updates) => updateAnswer(answer.id, updates)}
      />
    </div>
  );
};

export default SavedAnswersPage;
