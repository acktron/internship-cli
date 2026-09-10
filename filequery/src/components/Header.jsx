import {Moon,Search,Sun} from 'lucide-react'
export default function Header({theme,toggleTheme}){return <header><label><Search size={16}/><input placeholder="Search..." aria-label="Search files"/></label><button className="icon-button" onClick={toggleTheme} aria-label="Toggle theme">{theme==='dark'?<Sun size={16}/>:<Moon size={16}/>}</button></header>}
