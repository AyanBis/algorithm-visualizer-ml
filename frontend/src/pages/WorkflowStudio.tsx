import { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { ReactFlow, Background, Controls, MiniMap, Handle, Position, addEdge, MarkerType, type Connection, SmoothStepEdge } from '@xyflow/react'
import { Panel } from '../components/ui/Panel'
import { cn, formatDuration, formatTimestamp } from '../lib/utils'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs'
import { ScrollArea } from '../components/ui/ScrollArea'
import { Separator } from '../components/ui/Separator'
import { 
  PlayIcon, 
  RotateCcwIcon, 
  MaximizeIcon, 
  ActivityIcon,
  AlertCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
  SkipForwardIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  ListIcon,
  SquareIcon,
} from 'lucide-react'
import { useExperimentSSE } from '../features/experiments/sse'
import { useExperimentStatus, useExperimentResults, useCancelExperiment } from '../features/experiments/hooks'
import type { WorkflowEvent, StageInfo } from '../types/api'

const WORKFLOW_STAGES: StageInfo[] = [
  { id: 'experiment_config', name: 'Experiment Configuration', description: 'Validate inputs and prepare experiment configuration', order: 1 },
  { id: 'dataset_loading', name: 'Dataset Loading', description: 'Load dataset and extract metadata', order: 2 },
  { id: 'data_validation', name: 'Data Validation', description: 'Check for NaN, infinite values, alignment, and model/dataset compatibility', order: 3 },
  { id: 'train_test_split', name: 'Train/Test Split', description: 'Create reproducible train/test partitions with stratification for classification', order: 4 },
  { id: 'preprocessing', name: 'Preprocessing', description: 'Fit scaler on training data only, transform both train and test sets', order: 5 },
  { id: 'model_initialization', name: 'Model Initialization', description: 'Instantiate model with selected hyperparameters', order: 6 },
  { id: 'model_training', name: 'Model Training', description: 'Fit model on training partition', order: 7 },
  { id: 'prediction', name: 'Prediction / Assignment', description: 'Generate predictions on test set or cluster assignments', order: 8 },
  { id: 'evaluation', name: 'Evaluation', description: 'Compute metrics (accuracy, precision, recall, F1, MAE, RMSE, R², inertia, silhouette)', order: 9 },
  { id: 'visualization', name: 'Visualization', description: 'Generate visualization data (decision boundary, confusion matrix, clusters, etc.)', order: 10 },
  { id: 'results_ready', name: 'Results Ready', description: 'Finalize experiment, compile results', order: 11 },
]

const STAGE_STATUS_COLORS = {
  pending: 'bg-muted/20 text-muted border-muted/30',
  running: 'bg-info-light text-info border-info/30 animate-pulse',
  completed: 'bg-success-light text-success border-success/30',
  failed: 'bg-error-light text-error border-error/30',
  skipped: 'bg-warning-light text-warning border-warning/30',
  cancelled: 'bg-muted/30 text-muted border-muted/30',
}

const STAGE_STATUS_ICONS = {
  pending: ActivityIcon,
  running: ActivityIcon,
  completed: CheckCircleIcon,
  failed: AlertCircleIcon,
  skipped: SkipForwardIcon,
  cancelled: XCircleIcon,
}

interface WorkflowNodeData {
  stage: StageInfo
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled'
  event?: WorkflowEvent
  onClick: () => void
  [key: string]: unknown
}

function WorkflowNode({ data }: { data: WorkflowNodeData }) {
  const { stage, status, event, onClick } = data
  const Icon = STAGE_STATUS_ICONS[status]
  const isRunning = status === 'running'
  
  return (
    <div className={cn(
      'workflow-node relative',
      'rounded-lg border-2 p-3 min-w-[200px] max-w-[280px]',
      'transition-all duration-300',
      'hover:shadow-lg hover:border-accent/50',
      STAGE_STATUS_COLORS[status]
    )} onClick={onClick}>
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-accent border-2 border-background" />
      
      <div className="flex items-start gap-2">
        <div className={cn(
          'flex items-center justify-center w-8 h-8 rounded-full shrink-0',
          status === 'running' ? 'animate-pulse' : ''
        )}>
          <Icon className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{stage.name}</p>
          <p className="text-xs text-muted/80 mt-0.5 truncate">{stage.description}</p>
          {event?.duration_ms && (
            <p className="text-xs font-mono text-muted mt-1">
              {formatDuration(event.duration_ms)}
            </p>
          )}
        </div>
        {isRunning && (
          <div className="w-2 h-2 bg-info rounded-full animate-pulse" aria-hidden="true" />
        )}
      </div>
      
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-accent border-2 border-background" />
    </div>
  )
}

function WorkflowEdge(props: any) {
  return (
    <SmoothStepEdge
      {...props}
      type="smoothstep"
      animated={true}
      markerEndOffset={10}
    />
  )
}

function StageDetailPanel({ 
  stage, 
  event, 
  onClose 
}: { 
  stage: StageInfo
  event: WorkflowEvent | null
  onClose: () => void
}) {
  if (!event) {
    return (
      <div className="p-4 text-center text-muted">
        <ActivityIcon className="h-12 w-12 mx-auto mb-4 opacity-30" />
        <p>Click a stage to view details</p>
      </div>
    )
  }

  const StatusIcon = STAGE_STATUS_ICONS[event.status]
  const statusColor = STAGE_STATUS_COLORS[event.status]

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={cn('flex items-center justify-center w-10 h-10 rounded-full', statusColor)}>
            <StatusIcon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold">{stage.name}</h3>
            <p className="text-sm text-muted">{stage.description}</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <ChevronDownIcon className="h-4 w-4" />
        </Button>
      </div>

      <Separator />

      <div className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-muted">Status</span>
            <Badge variant={event.status === 'completed' ? 'success' : event.status === 'failed' ? 'destructive' : event.status === 'running' ? 'info' : event.status === 'skipped' ? 'warning' : 'secondary'} className="mt-1">
              {event.status}
            </Badge>
          </div>
          <div>
            <span className="text-muted">Sequence</span>
            <p className="font-mono mt-1">#{event.sequence}</p>
          </div>
          <div>
            <span className="text-muted">Started</span>
            <p className="font-mono mt-1">{event.started_at ? formatTimestamp(event.started_at) : '—'}</p>
          </div>
          <div>
            <span className="text-muted">Completed</span>
            <p className="font-mono mt-1">{event.completed_at ? formatTimestamp(event.completed_at) : '—'}</p>
          </div>
          <div className="col-span-2">
            <span className="text-muted">Duration</span>
            <p className="font-mono mt-1">{event.duration_ms ? formatDuration(event.duration_ms) : '—'}</p>
          </div>
        </div>

        {event.message && (
          <div>
            <span className="text-muted">Message</span>
            <p className="mt-1">{event.message}</p>
          </div>
        )}

        {event.error && (
          <div className="bg-error-light border border-error/30 rounded p-3">
            <span className="text-muted">Error</span>
            <p className="mt-1 text-error text-sm">{event.error}</p>
          </div>
        )}

        {Object.keys(event.details || {}).length > 0 && (
          <div>
            <span className="text-muted">Details</span>
            <pre className="mt-1 text-xs bg-muted p-2 rounded overflow-auto max-h-40">
              {JSON.stringify(event.details, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

function Timeline({ events, selectedEventId, onSelectEvent }: { 
  events: WorkflowEvent[]
  selectedEventId: string | null
  onSelectEvent: (id: string) => void
}) {
  return (
    <div className="space-y-3 max-h-96 overflow-y-auto">
      {events.length === 0 ? (
        <p className="text-center text-muted py-8">No events yet</p>
      ) : (
        events.map((event) => {
          const stage = WORKFLOW_STAGES.find(s => s.id === event.stage_id)
          const isSelected = selectedEventId === event.event_id
          const Icon = STAGE_STATUS_ICONS[event.status]
          
          return (
            <div
              key={event.event_id}
              onClick={() => onSelectEvent(event.event_id)}
              className={cn(
                'flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors',
                isSelected ? 'bg-accent-light border border-accent' : 'hover:bg-border/50'
              )}
            >
              <div className={cn(
                'flex items-center justify-center w-8 h-8 rounded-full shrink-0',
                STAGE_STATUS_COLORS[event.status]
              )}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="font-medium text-sm truncate">{stage?.name || event.stage_name}</p>
                  <Badge variant={event.status === 'completed' ? 'success' : event.status === 'failed' ? 'destructive' : event.status === 'running' ? 'info' : 'secondary'} className="text-xs">
                    {event.status}
                  </Badge>
                </div>
                <p className="text-xs text-muted truncate">{event.message}</p>
                {event.duration_ms && (
                  <p className="text-xs font-mono text-muted mt-0.5">{formatDuration(event.duration_ms)}</p>
                )}
              </div>
              <div className="text-right">
                <p className="text-xs text-muted">{event.timestamp ? formatTimestamp(event.timestamp) : ''}</p>
                {event.duration_ms && (
                  <p className="text-xs font-mono text-muted">{formatDuration(event.duration_ms)}</p>
                )}
              </div>
            </div>
          )
        })
      )}
    </div>
  )
}

function FlowControls({ 
  experimentId, 
  status, 
  isTraining,
  onStart,
  onCancel,
  onRerun,
  onReset,
  onFitView,
  onToggleAutoFollow,
  autoFollow,
  onToggleEventLog,
  showEventLog,
}: {
  experimentId: string | null
  status: string | null
  isTraining: boolean
  onStart: () => void
  onCancel: () => void
  onRerun: () => void
  onReset: () => void
  onFitView: () => void
  onToggleAutoFollow: () => void
  autoFollow: boolean
  onToggleEventLog: () => void
  showEventLog: boolean
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 p-4 border-t border-border bg-background/50 backdrop-blur-sm">
      <div className="flex items-center gap-2">
        {!experimentId && (
          <Button onClick={onStart} size="lg">
            <PlayIcon className="mr-2 h-4 w-4" aria-hidden="true" />
            Start Experiment
          </Button>
        )}
        {experimentId && isTraining && (
          <Button variant="destructive" onClick={onCancel} size="lg">
            <SquareIcon className="mr-2 h-4 w-4" aria-hidden="true" />
            Cancel
          </Button>
        )}
        {experimentId && !isTraining && status === 'completed' && (
          <Button variant="outline" onClick={onRerun} size="lg">
            <RotateCcwIcon className="mr-2 h-4 w-4" aria-hidden="true" />
            Rerun
          </Button>
        )}
        {experimentId && (
          <Button variant="ghost" onClick={onReset} size="lg">
            <RotateCcwIcon className="mr-2 h-4 w-4" aria-hidden="true" />
            Reset
          </Button>
        )}
      </div>
      
      <div className="flex-1" />
      
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={onFitView}>
          <MaximizeIcon className="h-4 w-4" aria-hidden="true" />
          Fit View
        </Button>
        <Button variant={autoFollow ? 'default' : 'outline'} size="sm" onClick={onToggleAutoFollow}>
          <ActivityIcon className="h-4 w-4" aria-hidden="true" />
          Auto-follow
        </Button>
        <Button variant={showEventLog ? 'default' : 'outline'} size="sm" onClick={onToggleEventLog}>
          <ListIcon className="h-4 w-4" aria-hidden="true" />
          Event Log
        </Button>
      </div>
    </div>
  )
}

export function WorkflowStudio() {
  const { experimentId } = useParams<{ experimentId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const [autoFollow, setAutoFollow] = useState(true)
  const [showEventLog, setShowEventLog] = useState(true)
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null)
  
  const { data: experimentStatus } = useExperimentStatus(experimentId, true)
  const { data: experimentResults } = useExperimentResults(experimentId)
  const cancelExperiment = useCancelExperiment()
  
  const [nodes, setNodes] = useState<any[]>([])
  const [edges, setEdges] = useState<any[]>([])
  const [events, setEvents] = useState<WorkflowEvent[]>([])
  
  // Initialize nodes and edges from workflow stages
  useEffect(() => {
    const initialNodes = WORKFLOW_STAGES.map((stage, index) => ({
      id: stage.id,
      type: 'workflowNode',
      position: { x: 0, y: index * 120 },
      data: {
        stage,
        status: 'pending' as const,
        onClick: () => {
          const stageEvents = events.filter(e => e.stage_id === stage.id)
          const latestEvent = stageEvents[stageEvents.length - 1] || null
          setSelectedEventId(latestEvent?.event_id || '')
        }
      },
    }))
    
    const initialEdges = WORKFLOW_STAGES.slice(0, -1).map((stage, index) => ({
      id: `edge-${stage.id}-${WORKFLOW_STAGES[index + 1].id}`,
      source: stage.id,
      target: WORKFLOW_STAGES[index + 1].id,
      type: 'workflowEdge',
      animated: false,
      markerEnd: { type: MarkerType.ArrowClosed, color: '#9CA3AF' },
    }))
    
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [])
  
  // SSE for real-time events
  useExperimentSSE({
    experimentId,
    enabled: !!experimentId,
    onEvent: (event) => {
      setEvents(prev => [...prev, event])
      
      // Update node status
      setNodes(prev => prev.map(node => {
        if (node.id === event.stage_id) {
          return {
            ...node,
            data: {
              ...node.data,
              status: event.status,
              event: event,
            }
          };
        }
        return node
      }));
      
      // Update edges animation
      if (event.event_type === 'stage_started') {
        setEdges(prev => prev.map(edge => {
          if (edge.target === event.stage_id) {
            return { ...edge, animated: true };
          }
          return edge
        }));
      } else if (event.event_type === 'stage_completed' || event.event_type === 'stage_failed') {
        setEdges(prev => prev.map(edge => {
          if (edge.target === event.stage_id) {
            return { ...edge, animated: false };
          }
          return edge
        }));
      }
      
      // Auto-follow active node
      if (autoFollow && (event.event_type === 'stage_started' || event.event_type === 'stage_completed')) {
        setTimeout(() => {
          reactFlowInstance?.fitView({ duration: 500, padding: 0.2 })
        }, 100);
      }
      
      // Refresh results when experiment completes
      if (event.event_type === 'experiment_completed' || event.event_type === 'experiment_failed') {
        queryClient.invalidateQueries({ queryKey: ['experiments', experimentId, 'results'] });
        queryClient.invalidateQueries({ queryKey: ['experiments', experimentId, 'status'] });
      }
    },
  })
  
  // Update node statuses from polled status
  useEffect(() => {
    if (experimentStatus && experimentStatus.current_stage) {
      setNodes(prev => prev.map(node => {
        const nodeStageOrder = WORKFLOW_STAGES.find(s => s.id === node.id)?.order || 0
        const currentStageOrder = WORKFLOW_STAGES.find(s => s.id === experimentStatus.current_stage)?.order || 0
        
        let newStatus = node.data.status
        if (nodeStageOrder < currentStageOrder) {
          newStatus = 'completed'
        } else if (nodeStageOrder === currentStageOrder && experimentStatus.status === 'running') {
          newStatus = 'running'
        } else if (experimentStatus.status === 'failed' && nodeStageOrder === currentStageOrder) {
          newStatus = 'failed'
        }
        
        if (newStatus !== node.data.status) {
          return { ...node, data: { ...node.data, status: newStatus } }
        }
        return node
      }))
    }
  }, [experimentStatus?.current_stage, experimentStatus?.status])
  
  const handleStart = useCallback(async () => {
    if (!experimentId) return
  }, [experimentId])
  
  const handleCancel = useCallback(async () => {
    if (!experimentId) return
    await cancelExperiment.mutateAsync(experimentId)
  }, [experimentId, cancelExperiment])
  
  const handleRerun = useCallback(() => {
    if (!experimentResults) return
    navigate(`/models?rerun=${experimentResults.experiment_id}`)
  }, [experimentResults, navigate])
  
  const handleReset = useCallback(() => {
    setNodes(prev => prev.map(node => ({
      ...node,
      data: { ...node.data, status: 'pending' as const, event: undefined }
    })))
    setEdges(prev => prev.map(edge => ({ ...edge, animated: false })))
    setEvents([])
    setSelectedEventId(null)
  }, [])
  
  const handleFitView = useCallback(() => {
    reactFlowInstance?.fitView({ duration: 500, padding: 0.2 })
  }, [reactFlowInstance])
  
  const handleOnConnect = useCallback((connection: Connection) => {
    setEdges(prev => addEdge(connection, prev))
  }, [])
  
  const getSelectedEvent = useCallback(() => {
    return events.find(e => e.event_id === selectedEventId) || null
  }, [events, selectedEventId])
  
  const selectedStage = WORKFLOW_STAGES.find(s => {
    const stageEvents = events.filter(e => e.stage_id === s.id)
    return stageEvents.some(e => e.event_id === selectedEventId)
  })
  
  const selectedEvent = events.find(e => e.event_id === selectedEventId) || null
  
  if (!experimentId) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Workflow Studio</h1>
            <p className="mt-1 text-muted">Select an experiment to view its workflow execution.</p>
          </div>
          <a href="/models" className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent rounded-md hover:bg-accent-hover transition-colors">
            <ExternalLinkIcon className="h-4 w-4" aria-hidden="true" />
            Train a Model
          </a>
        </div>
        <Card>
          <CardContent className="py-12 text-center">
            <ActivityIcon className="h-12 w-12 text-muted mx-auto mb-4" aria-hidden="true" />
            <h3 className="text-lg font-semibold text-foreground">No Experiment Selected</h3>
            <p className="mt-2 text-muted max-w-md mx-auto">
              Go to the Model Explorer, train a model, then click "View Workflow" to see the execution flow here.
            </p>
            <a href="/models" className="inline-flex items-center justify-center gap-2 mt-4 px-4 py-2 text-sm font-medium text-white bg-accent rounded-md hover:bg-accent-hover transition-colors">
              <ExternalLinkIcon className="h-4 w-4" aria-hidden="true" />
              Train a Model
            </a>
          </CardContent>
        </Card>
      </div>
    )
  }
  
  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 border-b border-border">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Workflow Studio</h1>
          <p className="mt-1 text-sm text-muted">
            Experiment: {experimentResults?.config.model_id} on {experimentResults?.config.dataset_id}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={experimentStatus?.status === 'running' ? 'info' : experimentStatus?.status === 'completed' ? 'success' : experimentStatus?.status === 'failed' ? 'destructive' : 'secondary'}>
            {experimentStatus?.status || 'unknown'}
          </Badge>
          {experimentResults && (
            <a href={`/models?exp=${experimentResults.experiment_id}`} className="inline-flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
              <ExternalLinkIcon className="mr-2 h-4 w-4" aria-hidden="true" />
              View Results
            </a>
          )}
        </div>
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Workflow Graph */}
        <div className="flex-1 relative">
          <ReactFlow
            ref={setReactFlowInstance}
            nodes={nodes}
            edges={edges}
            onConnect={handleOnConnect}
            nodeTypes={{ workflowNode: WorkflowNode }}
            edgeTypes={{ workflowEdge: WorkflowEdge }}
            fitView={false}
            panOnScroll={true}
            panOnScrollSpeed={0.5}
            minZoom={0.1}
            maxZoom={2}
            defaultViewport={{ x: 50, y: 50, zoom: 1 }}
            className="h-full w-full"
          >
            <Background color="#E5E7EB" gap={20} />
            <Controls 
              showZoom={true} 
              showFitView={true} 
              showInteractive={true}
            />
            <MiniMap 
              nodeColor={(node) => {
                const status = node.data?.status
                const colors = {
                  pending: '#9CA3AF',
                  running: '#0284C7',
                  completed: '#059669',
                  failed: '#DC2626',
                  skipped: '#D97706',
                  cancelled: '#9CA3AF',
                }
                return colors[status as keyof typeof colors] || '#9CA3AF'
              }}
            />
          </ReactFlow>
        </div>
        
        {/* Side Panel */}
        <div className={cn(
          'w-96 border-l border-border bg-background flex flex-col transition-all duration-300',
          showEventLog ? 'translate-x-0' : 'translate-x-full'
        )}>
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h3 className="font-semibold">Event Log</h3>
            <Button variant="ghost" size="icon" onClick={() => setShowEventLog(false)}>
              <ChevronRightIcon className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
          <ScrollArea className="flex-1">
            <Timeline 
              events={events} 
              selectedEventId={selectedEventId}
              onSelectEvent={setSelectedEventId}
            />
          </ScrollArea>
        </div>
        
        {/* Detail Panel */}
        <Panel position="bottom" className="h-[300px] border-t border-border bg-background">
          <StageDetailPanel
            stage={selectedStage || WORKFLOW_STAGES[0]}
            event={selectedEvent}
            onClose={() => {
              setSelectedEventId(null)
            }}
          />
        </Panel>
      </div>
      
      {/* Flow Controls */}
      <FlowControls
        experimentId={experimentId}
        status={experimentStatus?.status || null}
        isTraining={experimentStatus?.status === 'running' || experimentStatus?.status === 'pending'}
        onStart={handleStart}
        onCancel={handleCancel}
        onRerun={handleRerun}
        onReset={handleReset}
        onFitView={handleFitView}
        onToggleAutoFollow={() => setAutoFollow(!autoFollow)}
        autoFollow={autoFollow}
        onToggleEventLog={() => setShowEventLog(!showEventLog)}
        showEventLog={showEventLog}
      />
    </div>
  )
}