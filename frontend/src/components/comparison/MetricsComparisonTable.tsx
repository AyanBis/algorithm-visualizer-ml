import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../ui/Table'
import { TrophyIcon } from 'lucide-react'
import { cn } from '../../lib/utils'
import { formatNumber, getAvailableMetrics } from '../../lib/ml-utils'

interface MetricsComparisonTableProps {
  results: any[]
  taskType?: string
  primaryMetric: string
}

export function MetricsComparisonTable({ 
  results, 
  taskType, 
  primaryMetric 
}: MetricsComparisonTableProps) {
  if (!results.length) return null
  
  const metrics = getAvailableMetrics(taskType)
  const isHigherBetter = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette'].includes(primaryMetric)
  
  // Find best value for primary metric
  const validResults = results.filter(r => r.results?.metrics?.[primaryMetric] !== undefined)
  const bestValue = results.filter(r => r.results?.metrics?.[primaryMetric] !== undefined).reduce((best, r) => {
    const val = r.results.metrics[primaryMetric] as number
    return primaryMetric === 'accuracy' || primaryMetric === 'precision' || primaryMetric === 'recall' || primaryMetric === 'f1' || primaryMetric === 'r2' || primaryMetric === 'silhouette'
      ? Math.max(best, val)
      : Math.min(best, val)
  }, results[0].results.metrics[primaryMetric] as number)
  
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
            )}
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
                  <TableCell key={m.value} className={`text-right font-mono ${value === bestValue ? 'font-bold text-accent' : ''}`}>
                    {value !== undefined ? formatNumber(value as number) : 'N/A'}
                    {value === bestValue && <span className="ml-1 h-4 w-4 text-accent inline">★</span>}
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
          <li>Models are evaluated on the exact same train/test split for fair comparison.</          <li>The ★ indicates the primary metric for ranking. Higher is better for accuracy, precision, recall, F1, R², silhouette. Lower is better for MAE, RMSE, inertia.</li>
          <li>Different metrics may favor different models. Consider your specific use case when selecting a model.</li>
        </ul>
      </div>
    </div>
  )
}