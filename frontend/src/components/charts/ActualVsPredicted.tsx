import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Line, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/utils';

interface ActualVsPredictedProps {
  data: { actual: number[]; predicted: number[] };
  className?: string;
}

export function ActualVsPredicted({ data, className }: ActualVsPredictedProps) {
  const actual = data.actual;
  const predicted = data.predicted;
  const data_points = actual.map((act, i) => ({ actual: act, predicted: predicted[i] }));
  const minVal = Math.min(...actual, ...predicted);
  const maxVal = Math.max(...actual, ...predicted);

  return (
    <div className={`w-full h-[400px]`}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[minVal, maxVal]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: 'Actual Values', position: 'insideBottom', offset: -20 }}
          />
          <YAxis
            type="number"
            domain={[minVal, maxVal]}
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: 'Predicted Values', position: 'insideLeft', offset: 20 }}
          />
          <Tooltip
            formatter={(value: number | undefined, name: string) => [value !== undefined ? value.toFixed(3) : '', name]}
            labelFormatter={(value: number | undefined) => value !== undefined ? value.toFixed(1) : ''}
          />
          <Legend />
          <Scatter
            name="Predictions"
            data={data_points}
            fill="#3b82f6"
            stroke="#3b82f6"
          >
            <Tooltip />
          </Scatter>
          <Line
            type="linear"
            dataKey="x"
            data={[
              { x: minVal, y: minVal },
              { x: maxVal, y: maxVal },
            ]}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeWidth={2}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}