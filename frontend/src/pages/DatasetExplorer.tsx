import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { cn, formatNumber } from '../lib/utils'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../components/ui/Table'
import { Select } from '../components/ui/Select'
import { DatabaseIcon, BarChart3Icon, GridIcon, ChevronRightIcon } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'

const CHART_COLORS = [
  '#4F46E5', '#059669', '#DC2626', '#D97706', '#0284C7',
  '#7C3AED', '#EA580C', '#0891B2', '#65A30D', '#9333EA',
]

export function DatasetExplorer() {
  const navigate = useNavigate()
  const { data: datasetsResponse, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  })
  
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(
    datasetsResponse?.datasets[0]?.id || ''
  )
  
  const { data: datasetDetail, isLoading: detailLoading } = useQuery({
    queryKey: ['datasets', selectedDatasetId],
    queryFn: () => api.getDataset(selectedDatasetId),
    enabled: !!selectedDatasetId,
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 bg-border/50 rounded w-1/4 animate-pulse" />
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardContent className="pt-6">
                <div className="h-4 bg-border/50 rounded w-3/4 animate-pulse" />
                <div className="mt-4 h-4 bg-border/50 rounded w-1/2 animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const dataset = datasetDetail?.info
  const preview = datasetDetail?.preview || []
  const statistics = datasetDetail?.statistics || []
  const targetDistribution = datasetDetail?.target_distribution || null

  if (!dataset) {
    return (
      <div className="text-center py-12">
        <DatabaseIcon className="h-12 w-12 text-muted mx-auto mb-4" aria-hidden="true" />
        <h2 className="text-xl font-semibold text-foreground">No dataset selected</h2>
        <p className="mt-2 text-muted">Select a dataset from the list to explore.</p>
      </div>
    )
  }

  const handleDatasetChange = (id: string) => {
    setSelectedDatasetId(id)
  }

  const handleGoToModels = () => {
    navigate('/models')
  }

  return (
    <div className="space-y-6">
      {/* Dataset Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Dataset Explorer</h1>
          <p className="mt-1 text-muted">Select and inspect a dataset to understand its structure and characteristics.</p>
        </div>
        <Select
          value={selectedDatasetId}
          onValueChange={handleDatasetChange}
          options={datasetsResponse?.datasets.map(d => ({ value: d.id, label: d.name })) || []}
          placeholder="Select a dataset"
          className="w-full sm:w-64"
        />
      </div>

      {/* Dataset Info Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted">Task Type</p>
                <p className="text-2xl font-bold text-foreground capitalize">{dataset.task_type}</p>
              </div>
              <DatabaseIcon className="h-10 w-10 text-accent/20" aria-hidden="true" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted">Samples</p>
                <p className="text-2xl font-bold text-foreground">{dataset.n_samples.toLocaleString()}</p>
              </div>
              <GridIcon className="h-10 w-10 text-accent/20" aria-hidden="true" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted">Features</p>
                <p className="text-2xl font-bold text-foreground">{dataset.n_features}</p>
              </div>
              <BarChart3Icon className="h-10 w-10 text-accent/20" aria-hidden="true" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted">
                  {dataset.task_type === 'classification' ? 'Classes' : dataset.task_type === 'regression' ? 'Target' : 'Target'}
                </p>
                <p className="text-2xl font-bold text-foreground">
                  {dataset.target_names ? dataset.target_names.length : dataset.task_type === 'regression' ? 'Continuous' : 'N/A'}
                </p>
              </div>
              <Badge variant="outline" className="h-auto px-3 py-1">{dataset.task_type}</Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Description */}
      <Card>
        <CardHeader>
          <CardTitle>Description</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-foreground">{dataset.description}</p>
        </CardContent>
      </Card>

      {/* Data Preview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Data Preview</CardTitle>
            <p className="text-sm text-muted">First 10 rows</p>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {preview[0] ? Object.keys(preview[0]).map((key) => (
                    <TableHead key={key}>{key}</TableHead>
                  )) : null}
                </TableRow>
              </TableHeader>
              <TableBody>
                {preview.map((row, rowIndex) => (
                  <TableRow key={rowIndex}>
                    {Object.values(row).map((value, cellIndex) => (
                      <TableCell key={cellIndex}>
                        {typeof value === 'number' ? formatNumber(value) : String(value)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Feature Statistics & Target Distribution */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Feature Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">Non-Null</TableHead>
                    <TableHead className="text-right">Mean</TableHead>
                    <TableHead className="text-right">Std</TableHead>
                    <TableHead className="text-right">Min</TableHead>
                    <TableHead className="text-right">Max</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {statistics.map((stat) => (
                    <TableRow key={stat.feature}>
                      <TableCell className="font-mono text-sm">{stat.feature}</TableCell>
                      <TableCell className="text-right">{stat.non_null}</TableCell>
                      <TableCell className="text-right">{formatNumber(stat.mean)}</TableCell>
                      <TableCell className="text-right">{formatNumber(stat.std)}</TableCell>
                      <TableCell className="text-right">{formatNumber(stat.min)}</TableCell>
                      <TableCell className="text-right">{formatNumber(stat.max)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {targetDistribution && (
          <Card>
            <CardHeader>
              <CardTitle>Target Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {targetDistribution.map((item) => (
                  <div key={item.class} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{item.class}</span>
                      <span className="text-muted">{item.count} ({item.percentage.toFixed(1)}%)</span>
                    </div>
                    <div className="h-2 bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent rounded-full transition-all"
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Visualizations */}
      <Card>
        <CardHeader>
          <CardTitle>Visualizations</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="distributions" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="distributions">Feature Distributions</TabsTrigger>
              <TabsTrigger value="correlations">Correlation Matrix</TabsTrigger>
            </TabsList>
            <TabsContent value="distributions">
              <FeatureDistributionsChart features={statistics} targetDistribution={targetDistribution} />
            </TabsContent>
            <TabsContent value="correlations">
              <CorrelationMatrixChart features={statistics} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4">
        <Button onClick={handleGoToModels}>
          <ChevronRightIcon className="mr-2 h-4 w-4" aria-hidden="true" />
          Train Models on This Dataset
        </Button>
        <Button variant="outline" onClick={() => navigate('/comparison')}>
          Compare Models
        </Button>
      </div>
    </div>
  )
}

function FeatureDistributionsChart({ features, targetDistribution }: { 
  features: { feature: string; mean: number; std: number; min: number; max: number }[]
  targetDistribution: { class: string; count: number; percentage: number }[] | null
}) {
  // Simple histogram-like visualization using bars
  return (
    <div className="space-y-4 mt-4">
      <p className="text-sm text-muted">
        {targetDistribution 
          ? 'Feature distributions shown as normalized range bars (min-max). Class colors would be shown in full implementation.' 
          : 'Feature ranges shown as normalized bars (min-max).'}
      </p>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {features.map((f, i) => (
          <div key={f.feature} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="font-mono text-muted">{f.feature}</span>
              <span className="text-xs text-muted">
                μ={formatNumber(f.mean)} σ={formatNumber(f.std)}
              </span>
            </div>
            <div className="h-3 bg-border rounded-full relative overflow-hidden">
              <div
                className="h-full bg-accent rounded-full"
                style={{ 
                  left: '0%', 
                  right: '0%',
                  position: 'absolute',
                  opacity: 0.3,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function CorrelationMatrixChart({ features }: { 
  features: { feature: string; mean: number; std: number; min: number; max: number }[]
}) {
  return (
    <div className="mt-4">
      <p className="text-sm text-muted">
        Correlation matrix would be displayed here using a heatmap visualization. 
        Requires actual correlation data from the backend.
      </p>
      <div className="aspect-square mt-4 bg-border/30 rounded-lg flex items-center justify-center">
        <span className="text-muted">Correlation Heatmap Placeholder</span>
      </div>
    </div>
  )
}

// Re-export formatNumber for use in components
function formatNumber(value: number): string {
  if (value === 0) return '0'
  if (Math.abs(value) < 0.001) return value.toExponential(2)
  return value.toFixed(3)
}