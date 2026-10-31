import { Link, useLocation, NavLink } from 'react-router-dom'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { LayoutIcon, DatabaseIcon, BrainIcon, GitCompareIcon, ActivityIcon } from 'lucide-react'

const navigation = [
  { name: 'Overview', href: '/overview', icon: LayoutIcon },
  { name: 'Datasets', href: '/datasets', icon: DatabaseIcon },
  { name: 'Models', href: '/models', icon: BrainIcon },
  { name: 'Comparison', href: '/comparison', icon: GitCompareIcon },
  { name: 'Workflow', href: '/workflow', icon: ActivityIcon, disabled: true },
]

export function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-8">
          <Link to="/overview" className="flex items-center gap-2" aria-label="Algorithm Visualizer ML Home">
            <BrainIcon className="h-6 w-6 text-accent" aria-hidden="true" />
            <span className="font-semibold text-lg text-foreground">Algorithm Visualizer ML</span>
          </Link>
          <nav className="hidden md:flex items-center gap-1" aria-label="Main navigation">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  cn(
                    'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-accent-light text-accent'
                      : 'text-muted hover:text-foreground hover:bg-border/50'
                  )
                }
                aria-current={({ isActive }) => isActive ? 'page' : undefined}
                disabled={item.disabled}
              >
                <span className="flex items-center gap-2">
                  <item.icon className="h-4 w-4" aria-hidden="true" />
                  {item.name}
                </span>
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <a href="https://github.com/AyanBis/algorithm-visualizer-ml" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-foreground border border-border rounded-md hover:bg-border/50 transition-colors">
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.536-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
              </svg>
              GitHub
            </a>
        </div>
      </div>
    </header>
  )
}

export function Sidebar() {
  const location = useLocation()
  
  return (
    <aside className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-30 lg:w-64 lg:flex lg:flex-col border-r border-border bg-background">
      <div className="flex h-16 items-center px-4 border-b border-border">
        <Link to="/overview" className="flex items-center gap-2" aria-label="Algorithm Visualizer ML Home">
          <BrainIcon className="h-6 w-6 text-accent" aria-hidden="true" />
          <span className="font-semibold text-lg text-foreground">Algorithm Visualizer ML</span>
        </Link>
      </div>
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto" aria-label="Main navigation">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-accent-light text-accent'
                  : 'text-muted hover:text-foreground hover:bg-border/50'
              )
            }
            aria-current={({ isActive }) => isActive ? 'page' : undefined}
            disabled={item.disabled}
          >
            <item.icon className="h-5 w-5 shrink-0" aria-hidden="true" />
            {item.name}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-border">
        <p className="text-xs text-muted text-center">
          Built for education
        </p>
      </div>
    </aside>
  )
}

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <div className="flex-1 flex">
        <Sidebar />
        <main className="flex-1 lg:ml-64 min-w-0">
          <div className="container mx-auto px-4 py-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}