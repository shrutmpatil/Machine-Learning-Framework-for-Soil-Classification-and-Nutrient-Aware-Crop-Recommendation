import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
// import { Home } from 'lucide-react'
import Home from './Home.jsx'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Home/>
      <div>
        
      </div>  
    </>
  )
}

export default App
