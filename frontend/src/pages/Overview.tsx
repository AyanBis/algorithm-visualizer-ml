import { cn } from '../lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { ArrowRightIcon, DatabaseIcon, BrainIcon, GitCompareIcon, SparklesIcon, ZapIcon, BookOpenIcon } from 'lucide-react'

const taskTypes = [
  {
    name: 'Classification',
    description: 'Predict categorical labels',
    icon: DatabaseIcon,
    color: 'text-blue-600 bg-blue-50',
    models: ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'],
    href: '/models',
  },
  {
    name: 'Regression',
    description: 'Predict continuous values',
    icon: BrainIcon,
    color: 'text-green-600 bg-green-50',
    models: ['Linear Regression'],
    href: '/models',
  },
  {
    name: 'Clustering',
    description: 'Group similar data points',
    icon: GitCompareIcon,
    color: 'text-purple-600 bg-purple-50',
    models: ['K-Means'],
    href: '/models',
  },
]

const features = [
  {
    title: 'Interactive Exploration',
    description: 'Explore datasets, configure algorithms, and visualize results in real-time.',
    icon: ZapIcon,
    color: 'text-accent bg-accent/10',
  },
  {
    title: 'Educational Explanations',
    description: 'Every algorithm includes: what it does, how it works, when to use it, and its limitations.',
    icon: BookOpenIcon,
    color: 'text-green-600 bg-green-50',
  },
  {
    title: 'Rich Visualizations',
    description: 'Decision boundaries, confusion matrices, regression plots, cluster visualizations, and more.',
    icon: SparklesIcon,
    color: 'text-purple-600 bg-purple-50',
  },
  {
    title: 'Fair Model Comparison',
    description: 'Compare multiple algorithms on identical train/test splits with metric tables and charts.',
    icon: GitCompareIcon,
    color: 'text-orange-600 bg-orange-50',
  },
]

export function Overview() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="space-y-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            Algorithm Visualizer ML
          </h1>
          <p className="mt-4 text-lg text-muted max-w-2xl">
            An interactive web application for learning and exploring machine learning algorithms 
            through hands-on visualization. Understand how ML algorithms work by exploring datasets, 
            configuring algorithms, training models, and inspecting results.
          </p>
        </div>
        <div className="flex flex-wrap gap-4">
          <a href="/datasets" className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent rounded-md hover:bg-accent-hover transition-colors">
            <ArrowRightIcon className="mr-2 h-4 w-4" aria-hidden="true" />
            Explore Datasets
          </a>
          <a href="/models" className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
            Train a Model
          </a>
        </div>
      </section>

      {/* Features */}
      <section>
        <h2 className="text-2xl font-bold text-foreground mb-6">Key Features</h2>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <Card key={feature.title} className="hover:shadow-md transition-shadow">
              <CardContent className="pt-6">
                <div className={cn('w-12 h-12 rounded-lg flex items-center justify-center mb-4', feature.color)}>
                  <feature.icon className="h-6 w-6" aria-hidden="true" />
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
                <p className="mt-2 text-sm text-muted">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Supported Task Types */}
      <section>
        <h2 className="text-2xl font-bold text-foreground mb-6">Supported ML Task Types</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {taskTypes.map((task) => (
            <Card key={task.name} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className={cn('w-10 h-10 rounded-lg flex items-center justify-center', task.color)}>
                    <task.icon className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{task.name}</CardTitle>
                    <p className="text-sm text-muted">{task.description}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-2">
                  {task.models.map((model) => (
                    <li key={model} className="text-sm text-foreground flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-border" aria-hidden="true" />
                      {model}
                    </li>
                  ))}
                </ul>
                <a href={task.href} className="inline-flex items-center justify-center w-full px-4 py-2 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
                  View Algorithms
                  <ArrowRightIcon className="ml-2 h-4 w-4" aria-hidden="true" />
                </a>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Quick Start */}
      <section>
        <h2 className="text-2xl font-bold text-foreground mb-6">Quick Start</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {[
            {
              step: '01',
              title: 'Explore Data',
              description: 'Select a dataset from the Datasets page. Inspect features, distributions, correlations, and target balance.',
              href: '/datasets',
            },
            {
              step: '02',
              title: 'Choose Model',
              description: 'Pick a compatible algorithm on the Models page. Configure hyperparameters and training settings.',
              href: '/models',
            },
            {
              step: '03',
              title: 'Train & Visualize',
              description: 'Click Train to run the experiment. Watch the workflow execute in real-time, then explore metrics and visualizations.',
              href: '/workflow',
            },
          ].map((step) => (
            <Card key={step.step} className="relative overflow-hidden">
              <div className="absolute top-0 right-0 w-12 h-12 text-accent/10 flex items-center justify-center">
                <span className="text-2xl font-bold">{step.step}</span>
              </div>
              <CardContent className="pt-6 relative">
                <h3 className="text-lg font-semibold">{step.title}</h3>
                <p className="mt-2 text-sm text-muted">{step.description}</p>
                <a href={step.href} className="inline-flex items-center justify-center gap-2 mt-4 px-4 py-2 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
                  Get Started
                  <ArrowRightIcon className="ml-2 h-4 w-4" aria-hidden="true" />
                </a>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Datasets Overview */}
      <section>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-foreground">Built-in Datasets</h2>
          <a href="/datasets" className="inline-flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
            View All
          </a>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[
            { name: 'Iris', task: 'Classification', samples: 150, features: 4, classes: 3 },
            { name: 'Breast Cancer Wisconsin', task: 'Classification', samples: 569, features: 30, classes: 2 },
            { name: 'Diabetes', task: 'Regression', samples: 442, features: 10, target: 'Continuous' },
            { name: 'Synthetic Regression', task: 'Regression', samples: 200, features: 5, target: 'Continuous' },
            { name: 'Synthetic Blobs', task: 'Clustering', samples: 300, features: 2, clusters: 4 },
          ].map((dataset) => (
            <Card key={dataset.name} className="hover:shadow-md transition-shadow">
              <CardContent className="pt-6">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-lg">{dataset.name}</h3>
                    <Badge variant="outline" className="mt-1">{dataset.task}</Badge>
                  </div>
                  <DatabaseIcon className="h-8 w-8 text-muted/30" aria-hidden="true" />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-muted">Samples</span>
                    <p className="font-medium">{dataset.samples}</p>
                  </div>
                  <div>
                    <span className="text-muted">Features</span>
                    <p className="font-medium">{dataset.features}</p>
                  </div>
                  <div>
                    <span className="text-muted">
                      {dataset.task === 'Clustering' ? 'Clusters' : dataset.task === 'Regression' ? 'Target' : 'Classes'}
                    </span>
                    <p className="font-medium">{dataset.classes ?? dataset.clusters ?? dataset.target}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  )
}