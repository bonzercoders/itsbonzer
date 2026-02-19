import { useEffect, useState } from 'react'
import { Outlet } from 'react-router-dom'

import Sidebar from '@/components/layout/Sidebar'
import { cn } from '@/lib/utils'

function AppLayout() {
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    const root = document.documentElement
    root.style.setProperty(
      '--app-sidebar-width',
      collapsed ? '75px' : '220px'
    )
    root.style.setProperty('--app-layout-gap', '12px')
  }, [collapsed])

  return (
    <div className="dark min-h-screen bg-[#191b1f] text-[#e2e6ea]">
      <div className="flex h-screen gap-3 p-3">
        <Sidebar
          collapsed={collapsed}
          onToggle={() => setCollapsed((value) => !value)}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <main
            className={cn(
              'flex min-h-0 flex-1 rounded-2xl border border-[#333] bg-[#191b1f] p-8',
              'shadow-[0_20px_60px_rgba(0,0,0,0.35)]'
            )}
          >
            <div
              className={cn(
                'relative h-full w-full rounded-2xl border border-[#242830] bg-[#1c1e22]',
              )}
            >
            <Outlet />
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}

export default AppLayout
