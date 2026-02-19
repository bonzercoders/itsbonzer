import { OpenRouter } from '@openrouter/sdk'
import type { ModelsListResponse } from '@openrouter/sdk/models'

export type ModelOption = {
  value: string
  label: string
}

export type ModelGroup = {
  label: string
  options: ModelOption[]
}

const TOKEN_OVERRIDES: Record<string, string> = {
  ai: 'AI',
  api: 'API',
  gpt: 'GPT',
  llm: 'LLM',
  mistralai: 'MistralAI',
  openrouter: 'OpenRouter',
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
  xai: 'xAI',
}

const METRIC_SUFFIXES = new Set(['b', 'k', 'm', 't'])

const openRouter = new OpenRouter({
  apiKey: import.meta.env.VITE_OPENROUTER_API_KEY,
})

let modelGroupsCache: ModelGroup[] | null = null
let pendingModelGroupsRequest: Promise<ModelGroup[]> | null = null

function formatToken(token: string): string {
  const normalized = token.trim()
  if (!normalized) {
    return ''
  }

  const lower = normalized.toLowerCase()
  const override = TOKEN_OVERRIDES[lower]
  if (override) {
    return override
  }

  if (/^\d+[a-z]+$/.test(lower)) {
    const [, amount, suffix] = lower.match(/^(\d+)([a-z]+)$/) ?? []
    if (amount && suffix && suffix.length === 1 && METRIC_SUFFIXES.has(suffix)) {
      return `${amount}${suffix.toUpperCase()}`
    }
    return lower
  }

  if (/^[a-z]+\d/.test(lower)) {
    const [, prefix, rest] = lower.match(/^([a-z]+)(.*)$/) ?? []
    if (!prefix) {
      return normalized
    }
    const formattedPrefix =
      prefix.length <= 3 ? prefix.toUpperCase() : `${prefix[0].toUpperCase()}${prefix.slice(1)}`
    return `${formattedPrefix}${rest}`
  }

  if (/^[a-z]+$/.test(lower)) {
    return `${lower[0].toUpperCase()}${lower.slice(1)}`
  }

  return normalized
}

function formatSlug(slug: string): string {
  return slug
    .split(/[-_]/g)
    .map((part) =>
      part
        .split('.')
        .map((chunk) => formatToken(chunk))
        .filter(Boolean)
        .join('.')
    )
    .filter(Boolean)
    .join(' ')
}

function toModelOption(id: string): { providerLabel: string; option: ModelOption } | null {
  const [providerSlug, ...modelParts] = id.split('/')
  if (!providerSlug || modelParts.length === 0) {
    return null
  }

  const providerLabel = formatSlug(providerSlug)
  const modelSlug = modelParts.join('/')

  return {
    providerLabel,
    option: {
      value: id,
      label: formatSlug(modelSlug),
    },
  }
}

function mapModelsToGroups(models: ModelsListResponse): ModelGroup[] {
  const groupedOptions = new Map<string, ModelOption[]>()
  const seenModelIds = new Set<string>()

  for (const model of models.data) {
    if (!model.id || seenModelIds.has(model.id)) {
      continue
    }

    seenModelIds.add(model.id)

    const result = toModelOption(model.id)
    if (!result) {
      continue
    }

    const options = groupedOptions.get(result.providerLabel)
    if (options) {
      options.push(result.option)
      continue
    }

    groupedOptions.set(result.providerLabel, [result.option])
  }

  return Array.from(groupedOptions.entries())
    .map(([label, options]) => ({
      label,
      options: options.sort((a, b) => a.label.localeCompare(b.label)),
    }))
    .sort((a, b) => a.label.localeCompare(b.label))
}

export async function fetchOpenRouterModelGroups(): Promise<ModelGroup[]> {
  if (modelGroupsCache) {
    return modelGroupsCache
  }

  if (!pendingModelGroupsRequest) {
    pendingModelGroupsRequest = openRouter.models
      .list()
      .then((response) => {
        modelGroupsCache = mapModelsToGroups(response)
        return modelGroupsCache
      })
      .finally(() => {
        pendingModelGroupsRequest = null
      })
  }

  return pendingModelGroupsRequest
}
