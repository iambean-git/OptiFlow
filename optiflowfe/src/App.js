// import logo from './logo.svg';
// import './App.css';

import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import Test from "./Test";
import GraphTest from "./GraphTest";
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

        <Route path="/login" element={<Signin />} />
        <Route path="/" element={<Main />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
      {/* </div> */}
    </BrowserRouter>
  );
}

export default App;
