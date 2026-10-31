import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList, Cell } from 'recharts';
import { cn } from '../../lib/utils';

interface SilhouetteChartProps {
  data: { silhouette_score: number; silhouette_samples: number[]; labels: number[]; X_2d: number[][] };
  className?: string;
}

export function SilhouetteChart({ data, className }: SilhouetteChartProps) {
  const silhouette_score = data.silhouette_score;
  const silhouette_samples = data.silhouette_samples;
  const labels = data.labels;

  // Group by cluster for the bar chart
  const clusterData = silhouette_samples.reduce((acc: Record<number, number[]>, val: number, i: number) => {
    const cluster = labels[i];
    if (!acc[cluster]) acc[cluster] = [];
    acc[cluster].push(val);
    return acc;
  }, {} as Record<number, number[]>);

  const clusters = Object.keys(clusterData).map(Number).sort((a, b) => a - b);
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div className={`w-full h-[400px] ${className || ''}`}>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={clusters} layout="vertical" margin={{ top: 20, right: 30, left: 80, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" tickFormatter={(val) => val.toFixed(2)} />
          <YAxis type="category" dataKey="cluster" width={80} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value: number | undefined) => [value !== undefined ? value.toFixed(3) : '', 'Silhouette']} />
          <Legend />
          {clusters.map((cluster, index) => (
            <Bar
              key={cluster}
              dataKey={`cluster_${cluster}`}
              stackId="a"
              fill={colors[index % colors.length]}
              radius={[0, 4, 4, 0]}
            >
              <LabelList dataKey={`cluster_${cluster}`} position="insideRight" offset={5} fill="white" fontSize={10} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}