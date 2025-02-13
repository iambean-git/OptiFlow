import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { RecoilRoot } from "recoil";
import { toast, ToastContainer } from "react-toastify";

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

import Admin from "./pages/Admin";
function App() {
  return (
    <BrowserRouter>
      <RecoilRoot>
        <ToastContainer
          position="bottom-center"
          toastClassName={() => "bg-transparent shadow-none p-0"}
          bodyClassName={() => "p-0 m-0"}
          closeButton={false} // 기본 닫기 버튼 제거 (커스텀 버튼만 남기기)
          hideProgressBar={false} // 프로그레스 바 표시
          progressClassName="custom-progress-bar"
        />

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

            <Route path="/admin" element={<Admin />} />

          </Route>


        </Routes>
      </RecoilRoot>
    </BrowserRouter>
  );
}

export default App;
