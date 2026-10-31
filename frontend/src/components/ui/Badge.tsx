import { cn } from '../../lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' | 'info'
}

export function Badge({ className, variant = 'default', children, ...props }: BadgeProps) {
  const variants = {
    default: 'bg-border/50 text-foreground',
    secondary: 'bg-muted/50 text-muted',
    destructive: 'bg-error-light text-error',
    outline: 'text-foreground border border-border',
    success: 'bg-success-light text-success',
    warning: 'bg-warning-light text-warning',
    info: 'bg-info-light text-info',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}