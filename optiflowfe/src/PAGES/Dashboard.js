import React, { useEffect, useState } from 'react';
import { useLocation } from "react-router-dom";
import NavBar from "../components/NavBar";
import DashWaterLevel from "../components/dashboard/DashWaterLevel";
import CustomSelectBox from '../components/CustomSelectBox';
import DashWaterInfo from '../components/dashboard/DashWaterInfo';
import DashOutputPrediction from '../components/dashboard/DashOutputPrediction';
import DashOutput from '../components/dashboard/DashOutput';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import FetchFailed from '../components/FetchFailed';
import { NowDate } from "../recoil/DateAtom";
import { useRecoilValue } from "recoil";
import { formatDate } from "../utils/dateUtils";
import { toast } from 'react-toastify';
import CustomToast from "../components/ui/CustomToast";

export default function Dashboard() {
  const location = useLocation();

  const [selected, setSelected] = useState({ label: "D 배수지", value: "D" });
  const [selectedModel, setSelectedModel] = useState("xgb");

  const [options, setOptions] = useState([]);

  const [loading, setLoading] = useState(false);
  const [isfetchFailed, setIsFetchFailed] = useState(false);

  const [section1Data, setSection1Data] = useState(null);
  const [section2Data, setSection2Data] = useState(null);
  const [section2Prediction, setSection2Prediction] = useState(null);
  const [section3Data, setSection3Data] = useState(null);
  const [section4Data, setSection4Data] = useState(null);
  const [waterDetailInfo, setWaterDetailInfo] = useState(null);

  const [error, setError] = useState(null); // 에러 상태 저장

  // const loadingSpinner = <div className='w-full h-full flex justify-center items-center'><img className="size-[10vw]" src='/images/loadingSpinner.gif' /></div>
  const todayDate = (useRecoilValue(NowDate));
  useEffect(() => {
    fetchData1st(formatDate(todayDate));
    fetchData2nd(formatDate(todayDate));
    // fetchData3rd(formatDate(todayDate));
    setSelectedModel("xgb");

    // ============= 💥 원하는 시간으로 패치해보고 싶을 때 ==================
    // const hours = "10";
    // fetchData(`2023-10-21T${hours}:00`);
  }, []);


  // 비밀번호 변경 후 이동한 경우 토스트 띄우기
  useEffect(() => {
    if (location.state?.passwordChanged) {
      toast(<CustomToast msg={[`비밀번호가 성공적으로 변경되었습니다.`]} type={"dark"} icon={"success"} />, {
        autoClose: 2000, // 2초 후 자동 닫힘
        position: "bottom-center"
      });
    }
  }, [location.state]);


  useEffect(() => {
    // console.log("🌊 [DashBoard] selected :", selected.value);
    if (!selectedModel) return;
    fetchData2nd(formatDate(todayDate));
    fetchData3rd(formatDate(todayDate));
  }, [selectedModel, selected]);

  useEffect(() => {
    options.sort((a, b) => a.value.localeCompare(b.value)); // value기준 오름차순 정렬
    // console.log("🌊 [DashBoard] options :", options);
  }, [options]);



  useEffect(() => {
    if (!waterDetailInfo) return;
    setSection2Data(waterDetailInfo[selected.value]);
  }, [waterDetailInfo]);


  const fetchData1st = async (date) => {
    console.log("🌊 [DashBoard] fetchData1st 실행 :");
    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

    try {
      const url = `http://10.125.121.226:8080/api/reservoirdata/${date}`;
      console.log("🌊 [DashBoard] fetch :");
      const response = await fetch(url, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId); // 응답이 오면 타이머 제거

      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

      const data = await response.json();
      console.log("🌊 [DashBoard] 수위 데이터 :", data);

      const section1_data = [];
      const ops = [];
      const detailInfo = {};

      data.map((item) => {
        section1_data.push({
          id: item.reservoirId.name,
          percentage: (item.height / item.reservoirId.height * 100).toFixed(1)
        });
        ops.push({ value: (item.reservoirId.name).toUpperCase(), label: (item.reservoirId.name).toUpperCase() + " 배수지" });
        detailInfo[(item.reservoirId.name).toUpperCase()] = {
          crtWaterHeight: item.height,
          height: item.reservoirId.height,
          capacity: item.reservoirId.capacity,
          waterVol: item.height * item.reservoirId.area,
          input: item.input,
        };
      });

      console.log("🌊 [DashBoard] section1_data 데이터 :", section1_data);

      setSection1Data(section1_data);
      setOptions(ops);
      setWaterDetailInfo(detailInfo);

    } catch (err) {
      console.error("❌ [DashBoard] fetchData1st 실패:", err);
      setIsFetchFailed(true);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log("🌊 [DashBoard] section2Data 데이터 :", section2Data);
  }, [section2Data]);

  const fetchData2nd = async (date) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

    //이전 데이터
    try {
      const url = `http://10.125.121.226:8080/api/reservoirdata/${selected.value.toLowerCase()}/${date}`;
      const resp = await fetch(url, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId); // 응답이 오면 타이머 제거

      if (!resp.ok) throw new Error(`HTTP error! Status: ${resp.status}`);

      const data = await resp.json();
      console.log("🌊 [DashBoard] 이전 데이터 :", data);
      setSection3Data(data);

    } catch (err) {
      console.error("❌ [DashBoard] fetchData2nd(이전 데이터) 실패:", err);
      setError(err.message);
    }
  }

  const fetchData3rd = async (date) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

    try {
      const url = `http://10.125.121.226:8080/api/predict/${selectedModel}/${selected.value.toLowerCase()}/${date}`;
      console.log("예측 데이터 패치 url :", url);
      const resp = await fetch(url, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId); // 응답이 오면 타이머 제거
      if (!resp.ok) {
        throw new Error(`HTTP error! Status: ${resp.status}`);
      }
      const data = await resp.json();
      console.log("🌊 [DashBoard] 예측 데이터 :", data);
      // console.log("🌊 [DashBoard] 1시간 후 유출량 예측값 :", data.prediction[0]);
      setSection4Data(data); // 성공 시 실행
      const hours = date.substr(11, 2);
      setSection2Prediction({ hour: hours, data: data.prediction[0] });

    } catch (error) {
      console.error("❌ [DashBoard] fetchData3rd(예측 데이터) 실패:", error);
      setError(error.message);
      setSection4Data(null);
    }
  }

  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <div className="w-full h-[100px] px-10 flex justify-between">
          <header className="w-full  flex justify-between items-end text-[#333333]">
            {/* 텍스트 */}
            <h1 className="text-4xl font-medium text-[#333]">실시간 모니터링</h1>
            {/* selectBox */}
            <CustomSelectBox options={options} selectLabel={selected.label} selectedOption={selected} setSelectedOption={setSelected} />
          </header>
        </div>
        <section className="px-10 pb-10 pt-6 w-full h-full">
          {
            loading ? <LoadingSpinner /> :
              isfetchFailed ? <FetchFailed msg={"대시보드"} />
                :
                <div className="w-full h-full rounded-lg flex flex-col">
                  <div className="h-1/2 w-full flex gap-4">
                    <section className="w-3/4 bg-white rounded-lg">
                      <DashWaterLevel data={section1Data} selected={selected} setSelected={setSelected} />
                    </section>

                    <section className="w-1/4 rounded-lg">
                      <DashWaterInfo data={section2Data} predictionData={section2Prediction} />
                    </section>
                  </div>

                  <div className="h-1/2 w-full flex pt-4 gap-4">
                    <section className="w-1/2 bg-white rounded-lg">
                      <DashOutput data={section3Data} />
                    </section>

                    <section className="w-1/2 bg-white rounded-lg">
                      {
                        section4Data ?
                          <DashOutputPrediction data={section4Data} setModel={setSelectedModel} />
                          :
                          <FetchFailed msg={"예측"} />
                      }
                    </section>
                  </div>
                </div>
          }
        </section>
      </div>
    </div>
  );
}
