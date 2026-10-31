import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../lib/api'
import { cn, formatNumber, formatDuration } from '../lib/utils'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Slider } from '../components/ui/Slider'
import { Checkbox } from '../components/ui/Checkbox'
import { Label } from '../components/ui/Label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/Table'
import { useForm, useFieldArray } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { 
  PlayIcon, 
  GitCompareIcon, 
  RefreshCwIcon,
  ChevronRightIcon,
  TrophyIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from 'lucide-react'
import { useCreateExperiment, useExperimentStatus, useExperimentResults, useCancelExperiment } from '../features/experiments/hooks'
import { useExperimentSSE } from '../features/experiments/sse'
import { MetricsComparisonTable } from '../components/comparison/MetricsComparisonTable'
import { BarComparisonChart } from '../components/comparison/BarComparisonChart'
import { RadarComparisonChart } from '../components/comparison/RadarComparisonChart'

const CHART_COLORS = [
  '#4F46E5', '#059669', '#DC2626', '#D97706', '#0284C7',
  '#7C3AED', '#EA580C', '#0891B2', '#65A30D', '#9333EA',
]

export function ModelComparison() {
  const { datasetId } = useParams()
  const queryClient = useQueryClient()
  
  // Load datasets
  const { data: datasetsResponse } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  })
  
  // Selected dataset
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasetId || datasetsResponse?.datasets[0]?.id || '')
  const selectedDataset = datasetsResponse?.datasets.find(d => d.id === selectedDatasetId)
  
  // Load models
  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: () => api.getModels(),
  })
  
  // Filter compatible models
  const compatibleModels = modelsResponse?.models.filter(m => m.task_type === selectedDataset?.task_type) || []
  
  // Selected models for comparison
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(
    compatibleModels.slice(0, 3).map(m => m.id)
  )
  
  // Training config
  const [testSize, setTestSize] = useState(0.2)
  const [randomState, setRandomState] = useState(42)
  const [scaleFeatures, setScaleFeatures] = useState(true)
  const [scalerType, setScalerType] = useState<'standard' | 'minmax'>('standard')
  
  // Form for per-model hyperparameters
  const modelConfigSchema = z.record(z.object({
    hyperparameters: z.record(z.union([z.string(), z.number(), z.boolean()]))
  }))
  
  const form = useForm({
    resolver: zodResolver(modelConfigSchema),
    defaultValues: {},
  })
  
  const { fields, append, remove } = useFieldArray({ control: form.control, name: 'models' })
  
  // Initialize form when selected models change
  const selectedModels = compatibleModels.filter(m => selectedModelIds.includes(m.id))
  
  // Experiment mutations
  const createExperiment = useCreateExperiment()
  const cancelExperiment = useCancelExperiment()
  
  // Comparison experiment state
  const [experimentIds, setExperimentIds] = useState<string[]>([])
  const [isComparing, setIsComparing] = useState(false)
  const [comparisonResults, setComparisonResults] = useState<any[]>([])
  
  // Poll for experiment results
  const experimentStatusQueries = experimentIds.map(id => 
    useExperimentStatus(id, isComparing)
  )
  
  const experimentResultsQueries = experimentIds.map(id => 
    useExperimentResults(id)
  )
  
  // SSE for first experiment (for live updates)
  useExperimentSSE({
    experimentId: experimentIds[0],
    onEvent: (event) => {
      if (event.event_type === 'experiment_completed' || event.event_type === 'experiment_failed') {
        // Check if all experiments are done
        const allDone = experimentStatusQueries.every(q => 
          q.data?.status === 'completed' || q.data?.status === 'failed'
        )
        if (allDone) {
          setIsComparing(false)
          collectResults()
        }
      }
    },
    enabled: !!experimentIds[0] && isComparing,
  })
  
  const collectResults = () => {
    const mapped = experimentResultsQueries.map((q, i) => ({
      model: selectedModels[i],
      experimentId: experimentIds[i],
      status: experimentStatusQueries[i].data?.status,
      results: q.data,
    })
    const filtered = mapped.filter((r) => r.results);
    setComparisonResults(filtered);
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const handleCompare = async () => {
    if (selectedModelIds.length < 2) return
    
    setIsComparing(true)
    setExperimentIds([])
    setComparisonResults([])
    
    const newExperimentIds: string[] = []
    
    for (const modelId of selectedModelIds) {
      const model = compatibleModels.find(m => m.id === modelId)
      const hyperparams = form.getValues(`models.${selectedModelIds.indexOf(modelId)}.hyperparameters`) || model?.default_params || {}
      
      try {
        const response = await createExperiment.mutateAsync({
          config: {
            dataset_id: selectedDatasetId,
            model_id: modelId,
            hyperparameters: hyperparams,
            test_size: testSize,
            random_state: randomState,
            scale_features: scaleFeatures,
            scaler_type: scalerType,
            viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? [0, 1] : null,
          },
        })
        newExperimentIds.push(response.experiment_id)
      } catch (error) {
        console.error(`Failed to create experiment for ${modelId}:`, error)
      }
    }
    
    setExperimentIds(newExperimentIds)
  }
  
  const handleCancelAll = async () => {
    for (const id of experimentIds) {
      await cancelExperiment.mutateAsync(id)
    }
    setIsComparing(false)
  }
  
  const handleRerun = () => {
    setExperimentIds([])
    setComparisonResults([])
    setIsComparing(false)
  }
  
  const allCompleted = experimentIds.length > 0 && experimentStatusQueries.every(q => 
    q.data?.status === 'completed' || q.data?.status === 'failed'
  )
  
  if (isLoading || !selectedDataset) {
    return <div className="space-y-4">Loading...</div>
  }
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Model Comparison</h1>
          <p className="mt-1 text-muted">
            Compare multiple models on the same dataset with identical train/test splits.
          </p>
        </div>
        <Select
          value={selectedDatasetId}
          onChange={(e) => {
            const id = e.target.value
            setSelectedDatasetId(id)
            const newDataset = datasetsResponse?.datasets.find(d => d.id === id)
            const newModels = modelsResponse?.models.filter(m => m.task_type === newDataset?.task_type) || []
            const newSelected = newModels.slice(0, 3).map(m => m.id)
            setSelectedModelIds(newSelected)
          }}
          options={datasetsResponse?.datasets.map(d => ({ value: d.id, label: d.name })) || []}
          placeholder="Select dataset"
          className="w-full sm:w-48"
        />
      </div>
      
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left Panel: Configuration */}
        <div className="lg:col-span-1 space-y-6">
          {/* Model Selection */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Select Models to Compare</CardTitle>
                <Badge variant="outline">{selectedDataset?.task_type}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedModelIds.length === 0 ? (
                <p className="text-sm text-muted text-center py-4">No models selected</p>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {selectedModels.map((model, index) => (
                    <div key={model.id} className="flex items-center justify-between p-2 bg-border/30 rounded">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{index + 1}</span>
                        <span className="font-medium">{model.name}</span>
                      </div>
                      <Button variant="ghost" size="icon" onClick={() => setSelectedModelIds(selectedModelIds.filter(id => id !== model.id))}>
                        ×
                      </Button>
                    </div>
                  ))}
                </div>
              )}
              <div className="pt-2 border-t">
                <Select
                  value=""
                  onChange={(e) => setSelectedModelIds([...selectedModelIds, e.target.value])}
                  options={compatibleModels.filter(m => !selectedModelIds.includes(m.id)).map(m => ({ value: m.id, label: m.name }))}
                  placeholder="Add another model"
                  disabled={selectedModelIds.length >= compatibleModels.length}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>
          
          {/* Shared Training Config */}
          <Card>
            <CardHeader>
              <CardTitle>Shared Training Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1">
                <Label>Test Size</Label>
                <Slider
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  defaultValue={testSize}
                  onChange={(e) => setTestSize(Number(e.target.value))}
                  showValue
                />
              </div>
              <div className="space-y-1">
                <Label>Random Seed</Label>
                <Input
                  type="number"
                  min={0}
                  max={10000}
                  defaultValue={randomState}
                  onChange={(e) => setRandomState(parseInt(e.target.value) || 0)}
                />
              </div>
              <Checkbox
                id="cmp-scale-features"
                checked={scaleFeatures}
                onChange={(e) => setScaleFeatures(e.target.checked)}
                label="Scale Features"
              />
              {scaleFeatures && (
                <Select
                  value={scalerType}
                  onChange={(e) => setScalerType(e.target.value as 'standard' | 'minmax')}
                  options={[
                    { value: 'standard', label: 'StandardScaler' },
                    { value: 'minmax', label: 'MinMaxScaler' },
                  ]}
                  placeholder="Select scaler"
                />
              )}
            </CardContent>
          </Card>
          
          {/* Per-Model Hyperparameters */}
          {selectedModels.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Per-Model Hyperparameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedModels.map((model, index) => (
                  <div key={model.id} className="border-t pt-4 border-border/50">
                    <h4 className="font-medium mb-3">{model.name}</h4>
                    <div className="grid gap-3 md:grid-cols-2">
                      {model.hyperparameters.map((hp) => (
                        <div key={hp.name} className="space-y-1">
                          <Label htmlFor={`${model.id}-${hp.name}`}>{hp.display_name}</Label>
                          {hp.param_type === 'int' && (
                            <Slider
                              id={`${model.id}-${hp.name}`}
                              min={hp.min_value ?? 0}
                              max={hp.max_value ?? 100}
                              step={hp.step ?? 1}
                              defaultValue={Number(form.getValues(`models.${index}.hyperparameters.${hp.name}`) ?? hp.default)}
                              onChange={(e) => form.setValue(`models.${index}.hyperparameters.${hp.name}`, Number(e.target.value), { shouldValidate: true })}
                            />
                          )}
                          {hp.param_type === 'float' && (
                            <div className="flex items-center gap-2">
                              <Slider
                                id={`${model.id}-${hp.name}`}
                                min={hp.min_value ?? 0}
                                max={hp.max_value ?? 1}
                                step={hp.step ?? 0.01}
                                defaultValue={Number(form.getValues(`models.${index}.hyperparameters.${hp.name}`) ?? hp.default)}
                                onChange={(e) => form.setValue(`models.${index}.hyperparameters.${hp.name}`, Number(e.target.value), { shouldValidate: true })}
                                className="flex-1"
                              />
                              <Input
                                type="number"
                                step={hp.step ?? 0.01}
                                min={hp.min_value}
                                max={hp.max_value}
                                defaultValue={form.getValues(`models.${index}.hyperparameters.${hp.name}`) ?? hp.default}
                                onChange={(e) => form.setValue(`models.${index}.hyperparameters.${hp.name}`, parseFloat(e.target.value), { shouldValidate: true })}
                                className="w-24"
                              />
                            </div>
                          )}
                          {hp.param_type === 'select' && (
                            <Select
                              value={String(form.getValues(`models.${index}.hyperparameters.${hp.name}`) ?? hp.default)}
                              onChange={(e) => form.setValue(`models.${index}.hyperparameters.${hp.name}`, e.target.value, { shouldValidate: true })}
                              options={hp.options?.map(o => ({ value: String(o), label: String(o) })) || []}
                              placeholder="Select"
                            />
                          )}
                          {hp.help_text && <p className="text-xs text-muted">{hp.help_text}</p>}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          
          {/* Compare Button */}
          <Card>
            <CardContent className="pt-6">
              <Button 
                onClick={handleCompare} 
                disabled={isComparing || selectedModelIds.length < 2}
                className="w-full"
                size="lg"
              >
                {isComparing ? (
                  <>
                    <PlayIcon className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                    Running Comparison...
                  </>
                ) : (
                  <>
                    <GitCompareIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                    Compare Models
                  </>
                )}
              </Button>
              {isComparing && (
                <Button variant="outline" className="w-full mt-2" onClick={handleCancelAll}>
                  Cancel All
                </Button>
              )}
              {!isComparing && comparisonResults.length > 0 && (
                <Button variant="outline" className="w-full mt-2" onClick={handleRerun}>
                  <RefreshCwIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                  Rerun Comparison
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* Right Panel: Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Progress */}
          {isComparing && experimentIds.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Comparison in Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {experimentIds.map((id, i) => {
                    const model = selectedModels[i]
                    const status = experimentStatusQueries[i]?.data
                    return (
                      <div key={id} className="flex items-center justify-between p-3 bg-border/30 rounded">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium w-8">{i + 1}</span>
                          <span className="font-medium">{model.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {status && (
                            <Badge variant={
                              status.status === 'running' ? 'info' :
                              status.status === 'completed' ? 'success' :
                              status.status === 'failed' ? 'destructive' :
                              'secondary'
                            }>
                              {status.status}
                            </Badge>
                          )}
                          {status?.progress !== undefined && (
                            <div className="w-32 h-2 bg-border rounded-full overflow-hidden">
                              <div
                                className="h-full bg-accent transition-all duration-300"
                                style={{ width: `${(status.progress || 0) * 100}%` }}
                              />
                            </div>
                          )}
                        </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Comparison Results */}
          {comparisonResults.length > 0 && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Comparison Results</CardTitle>
                    <div className="flex items-center gap-2">
                      <Button variant="outline" size="sm" onClick={handleRerun}>
                        <RefreshCwIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                        Rerun
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Metric Selector */}
                  <div className="mb-4">
                    <Label htmlFor="primary-metric">Primary Metric</Label>
                    <Select
                      id="primary-metric"
                      value={primaryMetric}
                      onChange={(e) => setPrimaryMetric(e.target.value)}
                      options={getAvailableMetrics(selectedDataset?.task_type)}
                      placeholder="Select metric"
                      className="w-full max-w-xs"
                    />
                  </div>
                  
                  {/* Metrics Table */}
                  <MetricsComparisonTable 
                    results={comparisonResults} 
                    taskType={selectedDataset?.task_type}
                    primaryMetric={primaryMetric}
                  />
                  
                  {/* Charts */}
                  <Tabs defaultValue="bar" className="w-full mt-6">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="bar">Bar Chart</TabsTrigger>
                      <TabsTrigger value="radar">Radar Chart</TabsTrigger>
                    </TabsList>
                    <TabsContent value="bar">
                      <BarComparisonChart 
                        results={comparisonResults} 
                        metric={primaryMetric}
                        taskType={selectedDataset?.task_type}
                      />
                    </TabsContent>
                    <TabsContent value="radar">
                      <RadarComparisonChart 
                        results={comparisonResults} 
                        taskType={selectedDataset?.task_type}
                      />
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          )}
          
          {/* Empty State */}
          {!isComparing && comparisonResults.length === 0 && experimentIds.length === 0 && (
            <Card>
              <CardContent className="py-12 text-center">
                <GitCompareIcon className="h-12 w-12 text-muted mx-auto mb-4" aria-hidden="true" />
                <h3 className="text-lg font-semibold text-foreground">Ready to Compare</h3>
                <p className="mt-2 text-muted max-w-md mx-auto">
                  Select 2 or more compatible models, configure shared training settings, and click 
                  <strong>Compare Models</strong>. Each model will be trained on the exact same train/test split
                  for a fair comparison.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

const [primaryMetric, setPrimaryMetric] = useState('accuracy')

function getAvailableMetrics(taskType?: string) {
  if (taskType === 'classification') {
    return [
      { value: 'accuracy', label: 'Accuracy' },
      { value: 'precision', label: 'Precision (weighted)' },
      { value: 'recall', label: 'Recall (weighted)' },
      { value: 'f1', label: 'F1 (weighted)' },
    ]
  }
  if (taskType === 'regression') {
    return [
      { value: 'mae', label: 'MAE' },
      { value: 'rmse', label: 'RMSE' },
      { value: 'r2', label: 'R²' },
    ]
  }
  if (taskType === 'clustering') {
    return [
      { value: 'inertia', label: 'Inertia' },
      { value: 'silhouette', label: 'Silhouette' },
    ]
  }
  return [{ value: 'accuracy', label: 'Accuracy' }]
}

function MetricsComparisonTable({ 
  results, 
  taskType, 
  primaryMetric 
}: { 
  results: any[] 
  taskType?: string
  primaryMetric: string
}) {
  if (!results.length) return null
  
  const metrics = getAvailableMetrics(taskType)
  const isHigherBetter = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette'].includes(primaryMetric)
  
  // Find best value for primary metric
  const validResults = results.filter(r => r.results?.metrics?.[primaryMetric] !== undefined)
  const bestValue = isHigherBetter
    ? Math.max(...validResults.map(r => r.results.metrics[primaryMetric] as number))
    : Math.min(...validResults.map(r => r.results.metrics[primaryMetric] as number))
  
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            {metrics.map(m => (
              <TableHead key={m.value} className="text-right">
                {m.label}
                {m.value === primaryMetric && (
                  <span className="ml-1 text-xs text-accent">★</span>
                )}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {results.map((result, i) => (
            <TableRow key={result.experimentId}>
              <TableCell className="font-medium">{result.model.name}</TableCell>
              {metrics.map(m => {
                const value = result.results?.metrics?.[m.value]
                const isBest = value !== undefined && value === bestValue
                return (
                  <TableCell key={m.value} className={cn('text-right font-mono', isBest && 'font-bold text-accent')}>
                    {value !== undefined ? formatNumber(value as number) : 'N/A'}
                    {isBest && <TrophyIcon className="ml-1 h-4 w-4 text-accent inline" aria-hidden="true" />}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="mt-4 p-3 bg-info-light text-info rounded-lg text-sm">
        <p className="font-medium">Note:</p>
        <ul className="list-disc list-inside mt-1 space-y-1">
          <li>Models are evaluated on the exact same train/test split for fair comparison.</li>
          <li>Different metrics may favor different models. Consider your specific use case when selecting a model.</li>
        </ul>
      </div>
    </div>
  )
}

function BarComparisonChart({ 
  results, 
  metric, 
  taskType 
}: { 
  results: any[] 
  metric: string
  taskType?: string
}) {
  const isHigherBetter = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette'].includes(metric)
  
  return (
    <div className="h-64 mt-4">
      <p className="text-sm text-muted mb-4 text-center">
        Bar chart visualization would be rendered here using Recharts.
        Models compared on {metric.toUpperCase()}.
      </p>
      <div className="h-full bg-border/30 rounded-lg flex items-center justify-center">
        <p className="text-muted">Bar Chart Placeholder</p>
      </div>
    </div>
  )
}

function RadarComparisonChart({ 
  results, 
  taskType 
}: { 
  results: any[] 
  taskType?: string
}) {
  const metrics = getAvailableMetrics(taskType)
  
  return (
    <div className="h-64 mt-4">
      <p className="text-sm text-muted mb-4 text-center">
        Radar chart visualization would be rendered here using Recharts.
        Multi-metric comparison across all models.
      </p>
      <div className="h-full bg-border/30 rounded-lg flex items-center justify-center">
        <p className="text-muted">Radar Chart Placeholder</p>
      </div>
    </div>
  )
}

export default ModelComparison;
