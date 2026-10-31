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
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { 
  PlayIcon, 
  BrainIcon, 
  ZapIcon, 
  RefreshCwIcon,
  ExternalLinkIcon,
} from 'lucide-react'
import { useExperimentSSE } from '../features/experiments/sse'
import { useCreateExperiment, useExperimentStatus, useExperimentResults, useCancelExperiment } from '../features/experiments/hooks'
import { 
  ConfusionMatrix, 
  ActualVsPredicted, 
  FeatureImportanceChart, 
  ResidualsChart, 
  RegressionLineChart, 
  ClusterChart, 
  SilhouetteChart, 
  DecisionBoundaryChart,
} from '../components/charts'

const CHART_COLORS = [
  '#4F46E5', '#059669', '#DC2626', '#D97706', '#0284C7',
  '#7C3AED', '#EA580C', '#0891B2', '#65A30D', '#9333EA',
]

function ExperimentResultsPanel({ 
  experimentResults, 
  selectedDataset, 
  vizFeatures, 
  handleRerun, 
  experimentId 
}: { 
  experimentResults: any
  selectedDataset: any
  vizFeatures: [number, number]
  handleRerun: () => void
  experimentId: string
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Results</CardTitle>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleRerun}>
                <RefreshCwIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                Rerun
              </Button>
              <a href={`/workflow/${experimentResults.experiment_id}`} className="inline-flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
                <ExternalLinkIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                View Workflow
              </a>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <MetricsDisplay metrics={experimentResults.metrics} taskType={selectedDataset?.task_type} />
          
          {experimentResults.visualizations && Object.values(experimentResults.visualizations).some(v => v) && (
            <VisualizationTabs 
              visualizations={experimentResults.visualizations} 
              taskType={selectedDataset?.task_type}
              targetNames={selectedDataset?.target_names}
              featureNames={selectedDataset?.feature_names}
              vizFeatures={vizFeatures}
            />
          )}
          
          {experimentResults.predictions.length > 0 && (
            <div className="mt-6">
              <h3 className="font-semibold mb-3">Predictions Preview</h3>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Actual</TableHead>
                      <TableHead>Predicted</TableHead>
                      {experimentResults.predictions[0]?.probabilities && <TableHead>Max Probability</TableHead>}
                      <TableHead>Match</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {experimentResults.predictions.slice(0, 20).map((pred: any, i: number) => (
                      <TableRow key={i}>
                        <TableCell>{i + 1}</TableCell>
                        <TableCell>{pred.actual !== null ? String(pred.actual) : 'N/A'}</TableCell>
                        <TableCell>{String(pred.predicted)}</TableCell>
                        {pred.probabilities && (
                          <TableCell>{Math.max(...pred.probabilities).toFixed(3)}</TableCell>
                        )}
                        <TableCell>
                          {pred.match === true ? '✅' : pred.match === false ? '❌' : '—'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
          
          <div className="mt-6">
            <h3 className="font-semibold mb-3">Training Details</h3>
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <dt className="text-muted">Model</dt>
              <dd className="font-medium">{experimentResults.config.model_id}</dd>
              <dt className="text-muted">Train Time</dt>
              <dd className="font-medium">{formatDuration(experimentResults.train_time_ms)}</dd>
              <dt className="text-muted">Total Time</dt>
              <dd className="font-medium">{formatDuration(experimentResults.total_time_ms)}</dd>
              <dt className="text-muted">Train Samples</dt>
              <dd className="font-medium">{experimentResults.config.hyperparameters}</dd>
            </dl>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function ModelExplorer() {
  const navigate = useNavigate()
  const { datasetId } = useParams()
  const queryClient = useQueryClient()
  
  const { data: datasetsResponse, isLoading: datasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  })
  
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasetId || datasetsResponse?.datasets[0]?.id || '')
  
  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: () => api.getModels(),
  })
  
  const selectedDataset = datasetsResponse?.datasets.find(d => d.id === selectedDatasetId)
  const compatibleModels = modelsResponse?.models.filter(m => m.task_type === selectedDataset?.task_type) || []
  
  const [selectedModelId, setSelectedModelId] = useState(compatibleModels[0]?.id || '')
  const selectedModel = compatibleModels.find(m => m.id === selectedModelId)
  
  const { data: modelDetail } = useQuery({
    queryKey: ['models', selectedModelId],
    queryFn: () => api.getModel(selectedModelId),
    enabled: !!selectedModelId,
  })
  
  const model = modelDetail?.model
  
  const hyperparamSchema = z.record(z.string(), z.union([z.string(), z.number(), z.boolean()]))
  const form = useForm({
    resolver: zodResolver(hyperparamSchema),
    defaultValues: model?.default_params || {},
  })
  
  const [testSize, setTestSize] = useState(0.2)
  const [randomState, setRandomState] = useState(42)
  const [scaleFeatures, setScaleFeatures] = useState(true)
  const [scalerType, setScalerType] = useState<'standard' | 'minmax'>('standard')
  const [vizFeatures, setVizFeatures] = useState<[number, number]>([0, 1])
  
  const createExperiment = useCreateExperiment()
  const cancelExperiment = useCancelExperiment()
  
  const [experimentId, setExperimentId] = useState<string | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  
  const { data: experimentStatus } = useExperimentStatus(experimentId ?? undefined, isTraining)
  const { data: experimentResults } = useExperimentResults(experimentId ?? undefined)
  
  const { isConnected } = useExperimentSSE({
    experimentId: experimentId ?? undefined,
    onEvent: (event) => {
      if (event.event_type === 'experiment_completed' || event.event_type === 'experiment_failed') {
        setIsTraining(false)
        queryClient.invalidateQueries({ queryKey: ['experiments', experimentId, 'results'] })
      }
    },
    enabled: !!experimentId && isTraining,
  })
  
  const handleTrain = async (data: Record<string, unknown>) => {
    if (!selectedDatasetId || !selectedModelId) return
    
    try {
      setIsTraining(true)
      const response = await createExperiment.mutateAsync({
        config: {
          dataset_id: selectedDatasetId,
          model_id: selectedModelId,
          hyperparameters: data,
          test_size: testSize,
          random_state: randomState,
          scale_features: scaleFeatures,
          scaler_type: scalerType,
          viz_features: selectedDataset?.n_features && selectedDataset.n_features >= 2 ? vizFeatures : null,
        },
      })
      setExperimentId(response.experiment_id)
    } catch (error) {
      setIsTraining(false)
      console.error('Training failed:', error)
    }
  }
  
  const handleCancel = async () => {
    if (experimentId) {
      await cancelExperiment.mutateAsync(experimentId)
      setIsTraining(false)
    }
  }
  
  const handleRerun = () => {
    if (experimentId) {
      setExperimentId(null)
      setIsTraining(false)
      form.reset(model?.default_params || {})
    }
  }
  
  if (datasetsLoading || !selectedDataset) {
    return <div className="space-y-4">Loading...</div>
  }
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Model Explorer</h1>
          <p className="mt-1 text-muted">
            Configure and train machine learning models on your selected dataset.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Select
            value={selectedDatasetId}
            onChange={(e) => {
              const id = e.target.value
              setSelectedDatasetId(id)
              const newDataset = datasetsResponse?.datasets.find(d => d.id === id)
              const newModels = modelsResponse?.models.filter(m => m.task_type === newDataset?.task_type) || []
              setSelectedModelId(newModels[0]?.id || '')
            }}
            options={datasetsResponse?.datasets.map(d => ({ value: d.id, label: d.name })) || []}
            placeholder="Select dataset"
            className="w-full sm:w-48"
          />
        </div>
      </div>
      
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Select Model</CardTitle>
                <Badge variant="outline">{selectedDataset?.task_type}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <Select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                options={compatibleModels.map(m => ({ value: m.id, label: m.name }))}
                placeholder="Select model"
                className="w-full mb-4"
              />
              {selectedModel && (
                <div className="space-y-2 text-sm">
                  <p className="text-muted">{selectedModel.description}</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedModel.supports_probability && <Badge variant="secondary">Probability</Badge>}
                    {selectedModel.supports_decision_boundary && <Badge variant="secondary">Decision Boundary</Badge>}
                    {selectedModel.supports_feature_importance && <Badge variant="secondary">Feature Importance</Badge>}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          
          {selectedModel && (
            <Card>
              <CardHeader>
                <CardTitle>About This Model</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>How It Works</Label>
                  <p className="mt-1 text-sm text-muted">{selectedModel.how_it_works}</p>
                </div>
                <div>
                  <Label>When to Use</Label>
                  <p className="mt-1 text-sm text-muted">{selectedModel.when_to_use}</p>
                </div>
                <div>
                  <Label>Limitations</Label>
                  <p className="mt-1 text-sm text-muted">{selectedModel.limitations}</p>
                </div>
              </CardContent>
            </Card>
          )}
          
          {selectedModel && (
            <Card>
              <CardHeader>
                <CardTitle>Hyperparameters</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(handleTrain)} className="space-y-4">
{selectedModel.hyperparameters.map((hp: any) => (
                    <div key={hp.name} className="space-y-1">
                      <Label htmlFor={hp.name}>{hp.display_name}</Label>
                      {hp.param_type === 'int' && (
                        <Slider
                          id={hp.name}
                          min={hp.min_value ?? 0}
                          max={hp.max_value ?? 100}
                          step={hp.step ?? 1}
                          defaultValue={Number(form.getValues(hp.name) ?? hp.default)}
                          onChange={(e) => form.setValue(hp.name, Number(e.target.value), { shouldValidate: true })}
                        />
                      )}
                      {hp.param_type === 'float' && (
                        <div className="flex items-center gap-2">
                          <Slider
                            id={hp.name}
                            min={hp.min_value ?? 0}
                            max={hp.max_value ?? 1}
                            step={hp.step ?? 0.01}
                            defaultValue={Number(form.getValues(hp.name) ?? hp.default)}
                            onChange={(e) => form.setValue(hp.name, Number(e.target.value), { shouldValidate: true })}
                            className="flex-1"
                          />
                          <Input
                            id={`${hp.name}-input`}
                            type="number"
                            step={hp.step ?? 0.01}
                            min={hp.min_value}
                            max={hp.max_value}
                            defaultValue={form.getValues(hp.name) ?? hp.default}
                            onChange={(e) => form.setValue(hp.name, parseFloat(e.target.value), { shouldValidate: true })}
                            className="w-24"
                          />
                        </div>
                      )}
                      {hp.param_type === 'select' && (
                        <Select
                          value={String(form.getValues(hp.name) ?? hp.default)}
                          onChange={(e) => form.setValue(hp.name, e.target.value, { shouldValidate: true })}
                          options={hp.options?.map(o => ({ value: String(o), label: String(o) })) || []}
                          placeholder="Select"
                        />
                      )}
                      {hp.help_text && <p className="text-xs text-muted">{hp.help_text}</p>}
                    </div>
                  ))}
                  <div className="pt-4 border-t">
                    <Button type="submit" disabled={isTraining || createExperiment.isPending} className="w-full" size="lg">
                      {isTraining ? (
                        <>
                          <PlayIcon className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                          Training...
                        </>
                      ) : (
                        <>
                          <PlayIcon className="mr-2 h-4 w-4" aria-hidden="true" />
                          Train Model
                        </>
                      )}
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          )}
          
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
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
                id="scale-features"
                checked={scaleFeatures}
                onChange={(e) => setScaleFeatures(e.target.checked)}
                label="Scale Features (StandardScaler recommended)"
              />
              {scaleFeatures && (
                <Select
                  value={scalerType}
                  onChange={(e) => setScalerType(e.target.value as 'standard' | 'minmax')}
                  options={[
                    { value: 'standard', label: 'StandardScaler (zero mean, unit variance)' },
                    { value: 'minmax', label: 'MinMaxScaler (0 to 1 range)' },
                  ]}
                  placeholder="Select scaler"
                />
              )}
            </CardContent>
          </Card>
          
          {selectedDataset && selectedDataset.n_features >= 2 && (
            <Card>
              <CardHeader>
                <CardTitle>Visualization Features</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted mb-4">
                  Select two features for 2D visualizations (decision boundary, clusters, etc.)
                </p>
                <div className="grid gap-2 grid-cols-2">
                  <Select
                    value={vizFeatures[0]}
                    onChange={(e) => setVizFeatures([parseInt(e.target.value), vizFeatures[1]])}
                    options={selectedDataset.feature_names.map((f: string, i: number) => ({ value: String(i), label: f }))}
                    placeholder="Feature X"
                  />
                  <Select
                    value={vizFeatures[1]}
                    onChange={(e) => setVizFeatures([vizFeatures[0], parseInt(e.target.value)])}
                    options={selectedDataset.feature_names.map((f: string, i: number) => ({ value: String(i), label: f }))}
                    placeholder="Feature Y"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        
        <div className="lg:col-span-2 space-y-6">
          {experimentId && (
            <Card className="border-accent">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Experiment Running</CardTitle>
                <div className="flex items-center gap-2">
                  <Badge variant={isConnected ? 'success' : 'warning'}>
                    {isConnected ? 'Live' : 'Connecting...'}
                  </Badge>
                  {isTraining && (
                    <Button variant="outline" size="sm" onClick={handleCancel}>
                      Cancel
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {experimentStatus && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted">Status</span>
                      <Badge variant={
                        experimentStatus.status === 'running' ? 'info' :
                        experimentStatus.status === 'completed' ? 'success' :
                        experimentStatus.status === 'failed' ? 'destructive' :
                        'secondary'
                      }>
                        {experimentStatus.status}
                      </Badge>
                    </div>
                    {experimentStatus.current_stage && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted">Current Stage</span>
                        <span className="text-sm font-medium">{experimentStatus.current_stage}</span>
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted">Progress</span>
                      <div className="w-48 h-2 bg-border rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent transition-all duration-300"
                          style={{ width: `${(experimentStatus.progress || 0) * 100}%` }}
                        />
                      </div>
                    </div>
                    {experimentStatus.started_at && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted">Elapsed</span>
                        <span className="text-sm font-mono">
                          {formatDuration(Date.now() - new Date(experimentStatus.started_at).getTime())}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          
          {experimentResults && (
            <ExperimentResultsPanel 
              experimentResults={experimentResults}
              selectedDataset={selectedDataset}
              vizFeatures={vizFeatures}
              handleRerun={handleRerun}
              experimentId={experimentResults.experiment_id}
            />
          )}
          
          {!experimentId && !isTraining && (
            <Card>
              <CardContent className="py-12 text-center">
                <BrainIcon className="h-12 w-12 text-muted mx-auto mb-4" aria-hidden="true" />
                <h3 className="text-lg font-semibold text-foreground">Ready to Train</h3>
                <p className="mt-2 text-muted max-w-md mx-auto">
                  Configure your model and training settings, then click <strong>Train Model</strong> to start the experiment.
                  You'll see the workflow execute in real-time with live metrics and visualizations.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

function MetricsDisplay({ metrics, taskType }: { metrics: Record<string, unknown>; taskType?: string }) {
  if (!metrics || Object.keys(metrics).length === 0) return null
  
  if (taskType === 'classification') {
    return (
      <div className="grid gap-4 md:grid-cols-4 mb-6">
        <MetricCard label="Accuracy" value={formatNumber(metrics.accuracy as number)} />
        <MetricCard label="Precision (weighted)" value={formatNumber(metrics.precision as number)} />
        <MetricCard label="Recall (weighted)" value={formatNumber(metrics.recall as number)} />
        <MetricCard label="F1 (weighted)" value={formatNumber(metrics.f1 as number)} />
      </div>
    )
  }
  
  if (taskType === 'regression') {
    return (
      <div className="grid gap-4 md:grid-cols-3 mb-6">
        <MetricCard label="MAE" value={formatNumber(metrics.mae as number)} />
        <MetricCard label="RMSE" value={formatNumber(metrics.rmse as number)} />
        <MetricCard label="R²" value={formatNumber(metrics.r2 as number)} />
      </div>
    )
  }
  
  if (taskType === 'clustering') {
    return (
      <div className="grid gap-4 md:grid-cols-3 mb-6">
        <MetricCard label="Clusters" value={String(metrics.n_clusters)} />
        <MetricCard label="Inertia" value={formatNumber(metrics.inertia as number)} />
        <MetricCard label="Silhouette" value={metrics.silhouette !== null ? formatNumber(metrics.silhouette as number) : 'N/A'} />
      </div>
    )
  }
  
  return null
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <p className="text-sm text-muted">{label}</p>
      <p className="text-2xl font-bold text-foreground mt-1">{value}</p>
    </div>
  )
}

function VisualizationTabs({ 
  visualizations, 
  taskType, 
  targetNames,
  featureNames,
  vizFeatures,
}: { 
  visualizations: any
  taskType?: string
  targetNames?: string[] | null
  featureNames?: string[]
  vizFeatures: [number, number]
}) {
  const tabs = []
  
  if (visualizations.decision_boundary) tabs.push('decision_boundary')
  if (visualizations.confusion_matrix) tabs.push('confusion_matrix')
  if (visualizations.feature_importance) tabs.push('feature_importance')
  if (visualizations.actual_vs_predicted) tabs.push('actual_vs_predicted')
  if (visualizations.residuals) tabs.push('residuals')
  if (visualizations.regression_line) tabs.push('regression_line')
  if (visualizations.clusters) tabs.push('clusters')
  if (visualizations.silhouette) tabs.push('silhouette')
  
  if (tabs.length === 0) return null
  
  return (
    <div className="mt-6">
      <Tabs defaultValue={tabs[0]} className="w-full">
        <TabsList className="grid w-full grid-cols-{tabs.length > 4 ? 4 : tabs.length}">
{tabs.map((tab: string) => (
            <TabsTrigger key={tab} value={tab}>
              {tab.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabs.map((tab: string) => (
          <TabsContent key={tab} value={tab}>
            <VisualizationContent 
              type={tab} 
              data={visualizations[tab]} 
              targetNames={targetNames}
              featureNames={featureNames}
              vizFeatures={vizFeatures}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}

function VisualizationContent({ 
  type, 
  data, 
  targetNames,
  featureNames,
  vizFeatures,
}: { 
  type: string
  data: any
  targetNames?: string[] | null
  featureNames?: string[]
  vizFeatures: [number, number]
}) {
  if (!data) return <p className="text-center text-muted py-8">No data available</p>
  
  switch (type) {
    case 'decision_boundary':
      return <DecisionBoundaryChart {...data} targetNames={targetNames} />
    case 'confusion_matrix':
      return <ConfusionMatrix matrix={data} labels={targetNames} />
    case 'feature_importance':
      return <FeatureImportanceChart data={data} />
    case 'actual_vs_predicted':
      return <ActualVsPredicted data={data} />
    case 'residuals':
      return <ResidualsChart data={data} />
    case 'regression_line':
      return <RegressionLineChart data={data} />
    case 'clusters':
      return <ClusterChart data={data} />
    case 'silhouette':
      return <SilhouetteChart data={data} />
    default:
      return <pre className="bg-muted p-4 rounded text-xs overflow-auto">{JSON.stringify(data, null, 2)}</pre>
}
}

export default ModelExplorer;