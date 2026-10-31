import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { Overview } from './pages/Overview'
import { DatasetExplorer } from './pages/DatasetExplorer'
import { ModelExplorer } from './pages/ModelExplorer'
import { ModelComparison } from './pages/ModelComparison'
import { WorkflowStudio } from './pages/WorkflowStudio'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route path="overview" element={<Overview />} />
        <Route path="datasets" element={<DatasetExplorer />} />
        <Route path="models" element={<ModelExplorer />} />
        <Route path="comparison" element={<ModelComparison />} />
        <Route path="workflow/:experimentId" element={<WorkflowStudio />} />
      </Route>
      <Route path="*" element={<Navigate to="/overview" replace />} />
    </Routes>
  )
}

export default App