import {
  Bot,
  ChevronLeft,
  ChevronRight,
  Home,
  MessageSquare,
  Mic,
  Search,
  Settings,
  Sparkles,
  Users,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'

import { cn } from '@/lib/utils'
import bonzerFavicon from '../../../bonzer_favicon.png'

type SidebarProps = {
  collapsed: boolean
  onToggle: () => void
}

const navItems = [
  { to: '/', label: 'Home', icon: Home, accent: 'text-[#4da3ff]' },
  { to: '/models', label: 'Models', icon: Sparkles, accent: 'text-[#5fd4ff]' },
  { to: '/chats', label: 'Chats', icon: MessageSquare, accent: 'text-[#f2c94c]' },
  { to: '/characters', label: 'Characters', icon: Users, accent: 'text-[#57e389]' },
  { to: '/agents', label: 'Agents', icon: Bot, accent: 'text-[#ff6b6b]' },
  { to: '/speech', label: 'Speech', icon: Mic, accent: 'text-[#d17bff]' },
  { to: '/settings', label: 'Settings', icon: Settings, accent: 'text-[#b6d600]' },
]

function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={cn(
        'flex h-full flex-col rounded-2xl border border-[#333] bg-[#1c1e22]/90',
        'shadow-[0_10px_30px_rgba(0,0,0,0.35)]',
        'transition-[width] duration-300 ease-in-out',
        collapsed ? 'w-[75px]' : 'w-[220px]'
      )}
    >
      <div className={cn('relative px-3 py-3', collapsed && 'px-2')}>
        <button
          className="absolute right-3 top-3 inline-flex h-7 w-7 items-center justify-center rounded-md border border-[#2f353d] bg-[#1a1c20] text-[#c7cbd1] transition hover:border-[#3d3f44] hover:text-white"
          onClick={onToggle}
          type="button"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <ChevronRight className="h-3.5 w-3.5" />
          ) : (
            <ChevronLeft className="h-3.5 w-3.5" />
          )}
        </button>
        <div
          className={cn(
            'flex items-center gap-3',
            collapsed && 'justify-center'
          )}
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[#2f353d] bg-[#121417]">
            <img
              src={bonzerFavicon}
              alt="Bonzer"
              className="h-7 w-7 rounded-full"
            />
          </div>
          {!collapsed && (
            <div className="text-sm font-semibold tracking-wide text-[#e6eaef]">
              aiChat
            </div>
          )}
        </div>
      </div>

      <div className={cn('px-3 pb-3', collapsed && 'px-2')}>
        {collapsed ? (
          <button
            type="button"
            className="flex h-9 w-full items-center justify-center rounded-lg border border-[#2f353d] bg-[#22252a] text-[#8e96a1]"
            aria-label="Search"
            title="Search"
          >
            <Search className="h-4 w-4" />
          </button>
        ) : (
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#7e8791]" />
            <input
              className="h-9 w-full rounded-lg border border-[#2f353d] bg-[#22252a] pl-9 text-sm text-[#d5d9de] placeholder:text-[#6c7480] focus:border-[#3d4652] focus:outline-none"
              placeholder="Search"
              type="text"
            />
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-2">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'group flex items-center rounded-lg px-3 py-2 text-sm font-medium transition',
                  collapsed ? 'justify-center' : 'gap-3',
                  isActive
                    ? 'bg-[#007acc] text-white shadow-[0_8px_18px_rgba(0,122,204,0.35)]'
                    : 'text-[#c6cbd2] hover:bg-[#242830]'
                )
              }
              title={item.label}
            >
              <Icon className={cn('h-4 w-4', item.accent)} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          )
        })}
      </nav>

      <div className={cn('px-3 pb-3', collapsed && 'px-2')}>
        <div
          className={cn(
            'flex items-center gap-3 rounded-xl border border-[#2f353d] bg-[#1a1c20] px-2 py-2',
            collapsed && 'justify-center'
          )}
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-full border border-[#2f353d] bg-[#0f1216]">
            <img
              src={bonzerFavicon}
              alt="Bonzer"
              className="h-6 w-6 rounded-full"
            />
          </div>
          {!collapsed && (
            <div>
              <div className="text-xs font-semibold text-[#e3e7ec]">aiChat</div>
              <div className="text-[11px] text-[#7d8792]">Connected</div>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
