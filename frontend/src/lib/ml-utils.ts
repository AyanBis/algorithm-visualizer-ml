export function getAvailableMetrics(taskType?: string) {
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

export function formatNumber(value: number): string {
  if (value === 0) return '0'
  if (Math.abs(value) < 0.001) return value.toExponential(2)
  return value.toFixed(3)
}