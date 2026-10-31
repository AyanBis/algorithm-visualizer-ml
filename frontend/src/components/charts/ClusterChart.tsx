import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/utils';

interface ClusterChartProps {
  data: { X_2d: number[][]; labels: number[]; centroids: number[][] | null; feature_names: [string, string] };
  className?: string;
}

export function ClusterChart({ data, className }: ClusterChartProps) {
  const X_2d = data.X_2d;
  const labels = data.labels;
  const centroids = data.centroids;
  const feature_names = data.feature_names;

  const data_points = X_2d.map((point, i) => ({
    x: point[0],
    y: point[1],
    cluster: labels[i],
  }));

  const centroidData = centroids ? centroids.map((c, i) => ({ x: c[0], y: c[1], cluster: i })) : [];

  const minX = Math.min(...X_2d.map(p => p[0]));
  const maxX = Math.max(...X_2d.map(p => p[0]));
  const minY = Math.min(...X_2d.map(p => p[1]));
  const maxY = Math.max(...X_2d.map(p => p[1]));

  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div className={`w-full h-[400px]`}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[minX - 0.5, maxX + 0.5]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: feature_names[0], position: 'insideBottom', offset: -20 }}
          />
          <YAxis
            type="number"
            domain={[minY - 0.5, maxY + 0.5]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: feature_names[1], position: 'insideLeft', offset: 20 }}
          />
          <Tooltip
            formatter={(value: number | undefined) => [value !== undefined ? value.toFixed(3) : '', '']}
            labelFormatter={(value: number | undefined) => value !== undefined ? value.toFixed(1) : ''}
          />
          <Legend />
          <Scatter
            name="Data Points"
            data={data_points}
            fill="#8884d8"
            stroke="#8884d8"
          >
            <Tooltip />
          </Scatter>
          {centroids && centroids.length > 0 && (
            <Scatter
              name="Centroids"
              data={centroidData.map((c, i) => ({ ...c, cluster: i }))}
              fill="black"
              stroke="black"
              shape={<circle cx={0} cy={0} r={8} fill="black" />}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}