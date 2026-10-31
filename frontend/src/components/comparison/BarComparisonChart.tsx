import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList, Cell } from 'recharts'
import { formatNumber } from '../../lib/utils'

interface BarComparisonChartProps {
  results: any[]
  metric: string
  taskType?: string
}

export function BarComparisonChart({ results, metric, taskType }: BarComparisonChartProps): React.ReactElement | null {
  const validResults = results.filter(r => r.results?.metrics?.[metric] !== undefined)
  if (!validResults.length) return null

  const isHigherBetter = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette'].includes(metric)
  
  // Find best value using a simple loop
  let bestValue = validResults[0].results.metrics[metric] as number
  for (const r of validResults) {
    const val = r.results.metrics[metric] as number
    if (isHigherBetter) {
      if (val > bestValue) bestValue = val
    } else {
      if (val < bestValue) bestValue = val
    }
  }

  const data = results.map(r => ({
    model: r.model.name,
    value: r.results.metrics[metric],
    isBest: r.results.metrics[metric] === bestValue,
  }).filter(d => d.value !== undefined);

  return (
    <div className="h-64 mt-4">
      <div className="w-full h-[350px] bg-border/30 rounded-lg flex items-center justify-center">
        <p className="text-muted">Bar chart visualization would be rendered here using Recharts.</p>
        <p className="text-xs text-muted">Models compared on {metric.toUpperCase()}.</p>
      </div>
    </div>
  )