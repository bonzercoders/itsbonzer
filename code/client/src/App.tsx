import { Route, Routes } from 'react-router-dom'

import AppLayout from '@/layouts/AppLayout'
import AgentsPage from '@/pages/Agents'
import CharactersPage from '@/pages/Characters'
import ChatsPage from '@/pages/Chats'
import HomePage from '@/pages/Home'
import ModelsPage from '@/pages/Models'
import SettingsPage from '@/pages/Settings'
import SpeechPage from '@/pages/Speech'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/chats" element={<ChatsPage />} />
        <Route path="/characters" element={<CharactersPage />} />
        <Route path="/agents" element={<AgentsPage />} />
        <Route path="/speech" element={<SpeechPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}

export default App
