import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Line, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/utils';

interface RegressionLineProps {
  data: { X: number[]; y_true: number[]; y_pred: number[]; feature_name: string; target_name: string };
  className?: string;
}

export function RegressionLineChart({ data, className }: RegressionLineProps) {
  const X = data.X;
  const y_true = data.y_true;
  const y_pred = data.y_pred;
  const feature_name = data.feature_name;
  const target_name = data.target_name;

  const data_points = X.map((x, i) => ({ X: x, y_true: y_true[i], y_pred: y_pred[i] }));
  const sortIdx = [...data_points].sort((a, b) => a.X - b.X);
  const sortedData = sortIdx;

  const minX = Math.min(...X);
  const maxX = Math.max(...X);
  const allY = [...y_true, ...y_pred];
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);

  return (
    <div className={`w-full h-[400px]`}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[minX, maxX]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: feature_name, position: 'insideBottom', offset: -20 }}
          />
          <YAxis
            type="number"
            domain={[minY, maxY]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: target_name, position: 'insideLeft', offset: 20 }}
          />
          <Tooltip
            formatter={(value: number | undefined, name: string) => [value !== undefined ? value.toFixed(3) : '', name]}
            labelFormatter={(value: number | undefined) => value !== undefined ? value.toFixed(1) : ''}
          />
          <Legend />
          <Scatter
            name="Actual"
            data={data_points}
            fill="#3b82f6"
            stroke="#3b82f6"
            customShape={<circle cx={0} cy={0} r={4} fill="#3b82f6" />}
          >
            <Tooltip />
          </Scatter>
          <Line
            type="linear"
            dataKey="y_pred"
            data={sortedData}
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            name="Predicted"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}