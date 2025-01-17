// import logo from './logo.svg';
// import './App.css';

import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import Test from "./Test";
import GraphTest from "./test/GraphTest";
import FlowTest from "./test/FlowTest";
import FreeTest from "./test/FreeTest";

import Signin from "./PAGES/Signin";
import Main from "./PAGES/Main";
import Dashboard from "./PAGES/Dashboard";

function App() {
  return (
    <BrowserRouter>
     {/* <div className="w-[100px] h-[100px] bg-red-50"> */}
        <Routes>
          <Route path="/test" element={<Test />} />
          <Route path="/graph" element={<GraphTest />} />

      <Routes>
        <Route path="/test" element={<Test />} />
        <Route path="/graph" element={<GraphTest />} />
        <Route path="/flow" element={<FlowTest />} />
        
        <Route path="/free" element={<FreeTest />} />

        <Route path="/login" element={<Signin />} />
        <Route path="/" element={<Main />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
      {/* </div> */}

    </BrowserRouter>
  );
}

export default App;
