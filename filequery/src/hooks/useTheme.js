import { useEffect, useState } from 'react'

const DEFAULT_THEME = 'dark'
const THEME_KEY = 'filequery-theme-v2'
const LEGACY_THEME_KEY = 'filequery-theme'
const MIGRATION_KEY = 'filequery-theme-migrated'

function getInitialTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY)
  if (savedTheme === 'dark' || savedTheme === 'light') return savedTheme

  // The original key was written automatically on first render, so it cannot
  // distinguish a deliberate choice from an old development default.
  if (!localStorage.getItem(MIGRATION_KEY)) {
    localStorage.removeItem(LEGACY_THEME_KEY)
    localStorage.setItem(MIGRATION_KEY, 'true')
  }

  return DEFAULT_THEME
}

export function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  const toggleTheme = () => {
    setTheme((currentTheme) => {
      const nextTheme = currentTheme === 'dark' ? 'light' : 'dark'
      localStorage.setItem(THEME_KEY, nextTheme)
      return nextTheme
    })
  }

  return [theme, toggleTheme]
}
