import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList, Cell } from 'recharts';
import { cn } from '../../lib/utils';

interface FeatureImportanceChartProps {
  data: { feature: string; importance: number }[];
  className?: string;
}

export function FeatureImportanceChart({ data, className }: FeatureImportanceChartProps) {
  // Sort by importance descending
  const sortedData = [...data].sort((a, b) => b.importance - a.importance);

  return (
    <div className={cn('w-full h-[400px]', className)}>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={sortedData} layout="vertical" margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" tickFormatter={(val) => val.toFixed(3)} />
          <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value: number | undefined) => [value !== undefined ? value.toFixed(4) : '', 'Importance']} />
          <Legend />
          <Bar
            dataKey="importance"
            fill="#3b82f6"
            radius={[0, 4, 4, 0]}
          >
            <LabelList dataKey="importance" position="insideRight" offset={5} fill="white" fontSize={10} formatter={(val) => val !== undefined ? val.toFixed(4) : ''} />
            <Cell fill="#3b82f6" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}