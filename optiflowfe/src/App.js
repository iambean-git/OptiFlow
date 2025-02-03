// import logo from './logo.svg';
// import './App.css';

import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import "./css/fonts/Freesentation.css";
import "./css/fonts/Pretendard.css";

import "./css/fonts/SUIT.css";

import Test from "./Test";
import GraphTest from "./test/GraphTest";
import FreeTest from "./test/FreeTest";
import MapTest from "./mapTest/MapTest";

import Signin from "./pages/Signin";
import Main from "./pages/Main";
import Dashboard from "./pages/Dashboard";
import WaterLevel from "./pages/WaterLevel";
import Regions from "./pages/Regions";

function App() {
  return (
    <BrowserRouter>

     {/* <div className="w-[100px] h-[100px] bg-red-50"> */}
      <Routes>
        <Route path="/test" element={<Test />} />
        <Route path="/graph" element={<GraphTest />} />
        
        <Route path="/free" element={<FreeTest />} />
        <Route path="/maptest" element={<MapTest />} />

        <Route path="/login" element={<Signin />} />
        <Route path="/" element={<Main />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/waterlevel" element={<WaterLevel />} />
        <Route path="/regions" element={<Regions />} />

      </Routes>
      {/* </div> */}
    </BrowserRouter>
  );
}

export default App;
