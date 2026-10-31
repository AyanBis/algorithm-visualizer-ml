import { XAxis, YAxis, CartesianGrid, Tooltip, Scatter, ResponsiveContainer, ScatterChart } from 'recharts';
import { cn } from '../../lib/utils';

interface DecisionBoundaryProps {
  xx: number[][];
  yy: number[][];
  Z: number[][];
  X_test_2d: number[][];
  y_test: number[];
  feature_names: [string, string];
  target_names?: string[] | null;
  className?: string;
}

export function DecisionBoundaryChart({ xx, yy, Z, X_test_2d, y_test, feature_names, target_names, className }: DecisionBoundaryProps) {
  // Create grid data for decision boundary visualization
  const gridData: { x: number; y: number; class: number }[] = [];
  for (let i = 0; i < xx.length; i += 2) { // Sample every 2nd point for performance
    for (let j = 0; j < xx[i].length; j += 2) {
      gridData.push({
        x: xx[i][j],
        y: yy[i][j],
        class: Z[i][j],
      });
    }
  }

  // Test points
  const testPoints = X_test_2d.map((point, i) => ({
    x: point[0],
    y: point[1],
    label: y_test[i],
  }));

  const uniqueClasses = [...new Set(y_test)];
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
  const classLabels = target_names || uniqueClasses.map(c => `Class ${c}`);

  return (
    <div className={`w-full h-[400px] ${className || ''}`}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: feature_names[0], position: 'insideBottom', offset: -20 }}
          />
          <YAxis
            type="number"
            tickFormatter={(val) => val.toFixed(1)}
            label={{ value: feature_names[1], position: 'insideLeft', offset: 20 }}
          />
          <Tooltip
            formatter={(value: number | undefined) => [value !== undefined ? value.toFixed(3) : '', '']}
            labelFormatter={(value: number | undefined) => value !== undefined ? value.toFixed(1) : ''}
          />
          <Legend />
          
          {/* Decision boundary background - using scatter with low opacity */}
          <Scatter
            name="Decision Boundary"
            data={gridData.slice(0, 500)} // Limit points for performance
            fill="#888"
            stroke="none"
            opacity={0.1}
            customShape={<circle cx={0} cy={0} r={1} fill="#888" />}
          />
          
          {/* Test points */}
          <Scatter
            name="Test Points"
            data={testPoints}
            fill="#8884d8"
            stroke="#8884d8"
          >
            <Tooltip />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}