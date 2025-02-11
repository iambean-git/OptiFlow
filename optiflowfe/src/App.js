import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { RecoilRoot } from "recoil";

import ProtectedRoute from "./components/ProtectedRoute"; // ProtectedRoute 추가

import "./css/fonts/Freesentation.css";
import "./css/fonts/Pretendard.css";

import "./css/fonts/SUIT.css";

import Test from "./Test";
import GraphTest from "./test/GraphTest";
import FreeTest from "./test/FreeTest";

import Signin from "./pages/Signin";
import NotLogined from "./pages/NotLogined";
import NotFound from "./pages/NotFound";
import Main from "./pages/Main";
import Dashboard from "./pages/Dashboard";
import WaterLevel from "./pages/WaterLevel";
import Regions from "./pages/Regions";

function App() {
  return (
    <BrowserRouter>
      <RecoilRoot>
        <Routes>
          <Route path="/*" element={<NotFound />} />

          <Route path="/test" element={<Test />} />
          <Route path="/graph" element={<GraphTest />} />
          <Route path="/free" element={<FreeTest />} />

          <Route path="/login" element={<Signin />} />
          <Route path="/unauthorized" element={<NotLogined />} />

          <Route path="/" element={<Main />} />

          {/* 🛑 로그인된 사용자만 접근 가능하게 설정 */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/waterlevel" element={<WaterLevel />} />
            <Route path="/regions" element={<Regions />} />
          </Route>

        </Routes>
      </RecoilRoot>
    </BrowserRouter>
  );
}

export default App;
