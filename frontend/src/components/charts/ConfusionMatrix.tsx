import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LabelList, Cell } from 'recharts';
import { cn } from '../../lib/utils';

interface ConfusionMatrixProps {
  matrix: number[][];
  labels?: string[];
  className?: string;
}

export function ConfusionMatrix({ matrix, labels, className }: ConfusionMatrixProps) {
  const labelsToUse = labels || matrix.map((_, i) => `Class ${i}`);

  // Transform matrix data for Recharts (vertical layout for confusion matrix)
  const data = matrix.map((row, i) => {
    const rowData: Record<string, number | string> = { actual: labelsToUse[i] };
    row.forEach((val, j) => {
      rowData[`pred_${j}`] = val;
    });
    return rowData;
  });

  const keys = matrix[0].map((_, j) => `pred_${j}`);
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div className={cn('w-full', className)}>
      <BarChart data={data} layout="vertical" width={500} height={400} margin={{ top: 20, right: 30, left: 80, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" tickFormatter={(val) => val.toString()} />
        <YAxis type="category" dataKey="actual" width={80} tick={{ fontSize: 12 }}>
        </YAxis>
        <Tooltip formatter={(value: number) => [value.toString(), '']} />
        <Legend />
        {keys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            stackId="a"
            fill={colors[index % colors.length]}
            radius={[0, 4, 4, 0]}
          >
            <LabelList dataKey={key} position="insideRight" offset={5} fill="white" fontSize={10} />
          </Bar>
        ))}
      </BarChart>
    </div>
  );
}