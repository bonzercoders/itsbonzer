type PagePlaceholderProps = {
  title: string
  description?: string
}

function PagePlaceholder({ title, description }: PagePlaceholderProps) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center text-center">
      <div className="text-xs uppercase tracking-[0.35em] text-[#6f7782]">
        Placeholder
      </div>
      <h2 className="mt-2 text-2xl font-semibold text-[#dfe3e8]">{title}</h2>
      <p className="mt-2 max-w-md text-sm text-[#8a919c]">
        {description ?? 'This page will be designed next.'}
      </p>
    </div>
  )
}

export default PagePlaceholder
