import React, { useEffect, useState } from 'react';
import NavBar from "../components/NavBar";
import DashWaterLevel from "../components/dashboard/DashWaterLevel";
import CustomSelectBox from '../components/CustomSelectBox';
import DashWaterInfo from '../components/dashboard/DashWaterInfo';
import DashOutputPrediction from '../components/dashboard/DashOutputPrediction';
import DashOutput from '../components/dashboard/DashOutput';

import { NowDate } from "../recoil/DateAtom";
import { useRecoilValue } from "recoil";
import { formatDate } from "../utils/dateUtils";

export default function Dashboard() {
  const [selected, setSelected] = useState({ label: "J 배수지", value: "J" });
  const [options, setOptions] = useState([]);

  const [loading, setLoading] = useState(false);

  const [section1Data, setSection1Data] = useState(null);
  const [section2Data, setSection2Data] = useState(null);
  const [section2Prediction, setSection2Prediction] = useState(null);
  const [section3Data, setSection3Data] = useState(null);
  const [section4Data, setSection4Data] = useState(null);
  const [waterDetailInfo, setWaterDetailInfo] = useState(null);

  const loadingSpinner = <div className='w-full h-full flex justify-center items-center'><img className="size-[8vw]" src='/images/loadingSpinner.gif' /></div>
  const todayDate = (useRecoilValue(NowDate));
  useEffect(() => {
    fetchData1st(formatDate(todayDate));
    fetchData(formatDate(todayDate));
    // ============= 💥 원하는 시간으로 패치해보고 싶을 때 ==================
    // const hours = "10";
    // fetchData(`2023-10-21T${hours}:00`);
  }, []);

  useEffect(() => {
    options.sort((a, b) => a.value.localeCompare(b.value)); // value기준 오름차순 정렬
    // console.log("🌊 [DashBoard] options :", options);
  }, [options]);

  useEffect(() => {
    // console.log("🌊 [DashBoard] selected :", selected);
    if (!selected) return;
    if (!waterDetailInfo) return;
    setSection2Data(waterDetailInfo[selected.value]);
  }, [selected]);

  useEffect(() => {
    if (!waterDetailInfo) return;
    // console.log("🌊 [DashBoard] waterDetailInfo :", waterDetailInfo);
    // console.log("🌊 [DashBoard] selected :", selected.value);
    setSection2Data(waterDetailInfo[selected.value]);
  }, [waterDetailInfo]);


  // section2Data 확인용
  // useEffect(()=>{
  //   console.log("🌊 [DashBoard] section2Data :",section2Data);
  // },[section2Data]);

  const fetchData1st = async (date) => {
    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

    try {
      const url = `http://10.125.121.226:8080/api/reservoirdata/${date}`;
      const response = await fetch(url, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId); // 응답이 오면 타이머 제거

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      // console.log("🌊 [DashBoard] 수위 데이터 :", data);

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

      setSection1Data(section1_data);
      setOptions(ops);
      setWaterDetailInfo(detailInfo);

    } catch (err) {
      console.error("❌ [DashBoard] fetchData1st 실패:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchData = async (date) => {
    // ============= 💥 원하는 시간으로 패치해보고 싶을 때 ==================
    // const hours = "14";
    // const url = `http://10.125.121.226:8080/api/reservoirdata/2023-10-21T${hours}:00`;

    // const url = `http://10.125.121.226:8080/api/reservoirdata/${date}`;

    // const resp = await fetch(url);
    // const data = await resp.json();
    // // console.log("🌊 [DashBoard] 수위 데이터 :", data);

    // const section1_data = [];
    // const ops = [];
    // const detailInfo = {};

    // data.map((item) => {
    //   section1_data.push({
    //     id: item.reservoirId.name,
    //     percentage: (item.height / item.reservoirId.height * 100).toFixed(1)
    //   });
    //   ops.push({ value: (item.reservoirId.name).toUpperCase(), label: (item.reservoirId.name).toUpperCase() + " 배수지" });
    //   detailInfo[(item.reservoirId.name).toUpperCase()] = {
    //     crtWaterHeight: item.height,
    //     height: item.reservoirId.height,
    //     capacity: item.reservoirId.capacity,
    //     waterVol: item.height * item.reservoirId.area,
    //     input: item.input,
    //   };
    // });
    // // console.log("section1_data : ", section1_data);
    // // console.log("detailInfo : ", detailInfo);

    // setSection1Data(section1_data);
    // setOptions(ops);
    // setWaterDetailInfo(detailInfo);

    //이전데이터
    const url3 = `http://10.125.121.226:8080/api/reservoirdata/j/${date}`;
    const resp3 = await fetch(url3);
    const data3 = await resp3.json();
    console.log("🌊 [DashBoard] 이전 데이터 :", data3);
    setSection3Data(data3);

    try {
      const url2 = `http://10.125.121.226:8080/api/predict/${date}`;
      const resp2 = await fetch(url2);
      if (!resp2.ok) {
        throw new Error(`HTTP error! Status: ${resp2.status}`);
      }
      const data2 = await resp2.json();
      console.log("🌊 [DashBoard] 예측 데이터 :", data2);
      console.log("🌊 [DashBoard] 예측 데이터 :", data2.prediction[0]);
      setSection4Data(data2); // 성공 시 실행
      const hours = date.substr(11, 2);
      setSection2Prediction({ hour: hours, data: data2.prediction[0] });

    } catch (error) {
      // fetch 실패 시 실행할 코드
      console.error("❌ [DashBoard] 데이터 가져오기 실패:", error);
      setSection4Data(null);  // 예를 들어, 데이터를 초기화하거나 에러 메시지를 표시할 수 있음
    }

  }

  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <div className="w-full h-[100px] px-10 flex justify-between">
          {/* 텍스트 */}
          <div className="w-full  flex justify-between items-end text-[#333333]">
            <h1 className="text-4xl font-medium text-[#333]">실시간 모니터링</h1>
            <CustomSelectBox options={options} selectLabel={selected.label} selectedOption={selected} setSelectedOption={setSelected} />

          </div>

          {/* 달력 */}
          {/* <div className="h-full relative min-w-72 ">
          </div> */}
        </div>
        <section className="px-10 pb-10 pt-6 w-full h-full">
          <div className="w-full h-full rounded-lg flex flex-col">
            {loading ? loadingSpinner :
              <div className="h-1/2 w-full flex gap-4">
                <section className="w-3/4 bg-white rounded-lg">
                  <DashWaterLevel data={section1Data} selected={selected} setSelected={setSelected} />
                </section>

                <section className="w-1/4 rounded-lg">
                  <DashWaterInfo data={section2Data} predictionData={section2Prediction} />
                </section>
              </div>
            }


            <div className="h-1/2 w-full flex pt-4 gap-4">
              <section className="w-1/2 bg-white rounded-lg">
                {/* <DashOutput data={section3Data} /> */}
                {loadingSpinner}
              </section>

              <section className="w-1/2 bg-white rounded-lg">
                {
                  section4Data ?
                    <DashOutputPrediction data={section4Data} />
                    :
                    <div className='w-full h-full text-gray-600 flex flex-col justify-center items-center'>
                      <span className='text-lg'>예측 데이터를 로드할 수 없습니다.</span>
                      <button className='px-4 py-1 mt-4 border border-gray-400 rounded-lg text-sm'
                        onClick={() => window.location.reload()}
                      >
                        다시 시도
                      </button>

                    </div>
                }

              </section>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
