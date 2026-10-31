import { ResponsiveContainer } from 'recharts'

interface RadarComparisonChartProps {
  results: any[]
  taskType?: string
}

export function RadarComparisonChart({ results, taskType }: RadarComparisonChartProps) {
  return (
    <div className="h-64 mt-4">
      <div className="w-full h-[350px] bg-border/30 rounded-lg flex items-center justify-center">
        <p className="text-muted">Radar Chart Placeholder</p>
      </div>
    </div>
  )
}