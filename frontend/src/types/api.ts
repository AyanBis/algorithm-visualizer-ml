export interface DatasetInfo {
  id: string
  name: string
  task_type: 'classification' | 'regression' | 'clustering'
  description: string
  n_samples: number
  n_features: number
  feature_names: string[]
  target_name: string | null
  target_names: string[] | null
}

export interface DatasetStatistics {
  feature: string
  type: string
  non_null: number
  mean: number
  std: number
  min: number
  max: number
}

export interface DatasetDetailResponse {
  info: DatasetInfo
  preview: Record<string, unknown>[]
  statistics: DatasetStatistics[]
  target_distribution: { class: string; count: number; percentage: number }[] | null
}

export interface HyperParameter {
  name: string
  display_name: string
  param_type: 'int' | 'float' | 'select'
  default: number | string | boolean
  min_value: number | null
  max_value: number | null
  step: number | null
  options: (string | number | boolean)[] | null
  help_text: string
}

export interface ModelInfo {
  id: string
  name: string
  task_type: 'classification' | 'regression' | 'clustering'
  description: string
  how_it_works: string
  when_to_use: string
  limitations: string
  hyperparameters: HyperParameter[]
  supports_probability: boolean
  supports_decision_boundary: boolean
  supports_feature_importance: boolean
  default_params: Record<string, unknown>
}

export interface ExperimentConfig {
  dataset_id: string
  model_id: string
  hyperparameters: Record<string, unknown>
  test_size: number
  random_state: number
  scale_features: boolean
  scaler_type: 'standard' | 'minmax'
  viz_features: [number, number] | null
}

export interface ExperimentCreateRequest {
  config: ExperimentConfig
}

export interface ExperimentCreateResponse {
  experiment_id: string
  status: string
  created_at: string
}

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface ExperimentStatusResponse {
  experiment_id: string
  status: ExperimentStatus
  config: ExperimentConfig
  current_stage: string | null
  progress: number
  started_at: string | null
  completed_at: string | null
  total_duration_ms: number | null
}

export interface StageTiming {
  stage_id: string
  stage_name: string
  duration_ms: number
  status: string
}

export interface VisualizationData {
  decision_boundary: DecisionBoundaryData | null
  feature_importance: FeatureImportanceItem[] | null
  confusion_matrix: number[][] | null
  actual_vs_predicted: { actual: number[]; predicted: number[] } | null
  residuals: { predicted: number[]; residuals: number[] } | null
  regression_line: { X: number[]; y_true: number[]; y_pred: number[]; feature_name: string; target_name: string } | null
  clusters: ClusterData | null
  silhouette: SilhouetteData | null
}

export interface DecisionBoundaryData {
  xx: number[][]
  yy: number[][]
  Z: number[][]
  X_test_2d: number[][]
  y_test: number[]
  feature_names: [string, string]
  target_names: string[] | null
}

export interface FeatureImportanceItem {
  feature: string
  importance: number
}

export interface ClusterData {
  X_2d: number[][]
  labels: number[]
  centroids: number[][] | null
  feature_names: [string, string]
}

export interface SilhouetteData {
  silhouette_score: number
  silhouette_samples: number[]
  labels: number[]
  X_2d: number[][]
}

export interface PredictionItem {
  actual: number | string | null
  predicted: number | string
  probabilities: number[] | null
  match: boolean | null
}

export interface ExperimentResultsResponse {
  experiment_id: string
  status: ExperimentStatus
  config: ExperimentConfig
  metrics: Record<string, unknown>
  visualizations: VisualizationData
  predictions: PredictionItem[]
  train_time_ms: number
  total_time_ms: number
  stage_timings: StageTiming[]
  warnings: string[]
}

export interface WorkflowEvent {
  experiment_id: string
  event_id: string
  event_type: 'stage_started' | 'stage_completed' | 'stage_failed' | 'stage_skipped' | 'experiment_completed' | 'experiment_failed'
  stage_id: string
  stage_name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled'
  sequence: number
  timestamp: string
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
  message: string
  details: Record<string, unknown>
  error: string | null
}

export interface StageInfo {
  id: string
  name: string
  description: string
  order: number
}