import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Line, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/utils';

interface ResidualsProps {
  data: { predicted: number[]; residuals: number[] };
  className?: string;
}

export function ResidualsChart({ data, className }: ResidualsProps) {
  const predicted = data.predicted;
  const residuals = data.residuals;
  const data_points = predicted.map((pred, i) => ({ predicted: pred, residual: residuals[i] }));
  const minPred = Math.min(...predicted);
  const maxPred = Math.max(...predicted);
  const minRes = Math.min(...residuals);
  const maxRes = Math.max(...residuals);

  return (
    <div className={`w-full h-[400px] ${className || ''}`}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[minPred, maxPred]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: 'Predicted Values', position: 'insideBottom', offset: -20 }}
          />
          <YAxis
            type="number"
            domain={[minRes, maxRes]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: 'Residuals', position: 'insideLeft', offset: 20 }}
          />
          <Tooltip 
            formatter={(value: number | undefined, name: string) => [value !== undefined ? value.toFixed(3) : '', name]}
            labelFormatter={(value: number | undefined) => value !== undefined ? value.toFixed(1) : ''}
          />
          <Legend />
          <Line
            type="linear"
            dataKey="x"
            data={[
              { x: minPred, y: 0 },
              { x: maxPred, y: 0 },
            ]}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeWidth={2}
          />
          <Scatter
            name="Residuals"
            data={data_points}
            fill="#3b82f6"
            stroke="#3b82f6"
          >
            <Tooltip />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}