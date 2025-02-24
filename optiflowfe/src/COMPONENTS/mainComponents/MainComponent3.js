import "../../css/videoStyle.css";
import { AiOutlineDashboard } from "react-icons/ai";
import { SlGraph } from "react-icons/sl";

import { useState, useEffect } from "react";
export default function MainComponent3({ isvisible }) {
  const [isAnimaged, setIsAnimated] = useState(false);
  const [isAnimaged2, setIsAnimated2] = useState(false);

  useEffect(() => {
    if (isvisible) {
      setTimeout(() => {
        setIsAnimated(true);
      }, 600); 
      setTimeout(() => {
        setIsAnimated2(true);
      }, 1300); 
    }
  }, [isvisible]);

  return (
    <div className="main-component-container flex justify-center">

      <video
        // src="/videos/수돗물.mp4"
        src="/videos/waterdrop2.mp4"
        loop
        muted
        autoPlay
        className="background-video opacity-30"
      />

      <div className="w-[70%] 2xl:w-[55%] h-full grid grid-rows-2 py-20 p-4 gap-10 ">

        {/* ==========  1번째 영역 [DASHBOARD] ========== */}
        <section className={`flex justify-between  ${isAnimaged ? "animationSlideUp" : "opacity-0"}`}>
          <div className="flex items-center justify-center h-full rounded-lg  ">
            <img src="/images/mainImg/cap_dashboard1.png" alt="dashboard"
              className="rounded-lg mr-2 h-[75%] shadow-lg" />
            <img src="/images/mainImg/cap_dashboard2.png" alt="dashboard"
              className="rounded-lg h-4/5  " />
          </div>

          <div className="flex flex-col justify-center mr-20">
            <p className="text-lg text-blue-500 font-semibold flex items-center">
              <AiOutlineDashboard className="mr-1" />
              DASHBOARD
            </p>
            <p className="text-3xl font-bold">
              실시간 배수지 정보를 한눈에
            </p>
            <p className="text-lg">

            </p>
            <div className="text-lg text-gray-500 mt-6">
              <p className="">
                실시간 배수지별 저수량 모니터링,
              </p>
              <p className="">
                과거 24시간 유입량과 유출량 정보,
              </p>
              <p className="">
                미래 24시간 예측 결과를 한 곳에서 볼 수 있습니다
              </p>
            </div>


          </div>
        </section>
        
        {/* ==========  2번째 영역 [visualization ] ========== */}
        <section className={`flex justify-between items-center  ${isAnimaged2 ? "animationSlideUp" : "opacity-0"}`}>
          <div className="flex flex-col justify-center ml-14">
            <p className="text-lg text-blue-500 font-semibold flex items-center">
              <SlGraph className="mr-1" />
              VISUALIZATION
            </p>
            <p className="text-3xl font-bold">
              기계학습 기반 물 소비량 예측
            </p>
            <p className="text-lg">

            </p>
            <div className="text-lg text-gray-500 mt-6">
              <p className="">
                AI 모델을 활용해 시간당 유출량을 예측하고,
              </p>
              <p className="">
                전기 요금 절약을 위한 개선된 운영 계획을 제공합니다.
              </p>
            </div>

          </div>
          <img src="/images/mainImg/cap_graph.png" alt="dashboard"
            className="rounded-lg ml-12 h-4/5 shadow-xl" />
        </section>

      </div>
    </div>
  );
}
