import "../../css/videoStyle.css"; 
import { useEffect, useRef, useState } from "react";

export default function MainComponent2({ isvisible }) {
  const [isAnimaged, setIsAnimated] = useState(false);

  useEffect(() => {
    if (isvisible) {
      setTimeout(() => {
        setIsAnimated(true);
      }, 200); // 300ms(0.3초) 후 애니메이션 실행
    } 
  }, [isvisible]);

  return (
    <div className="inner scorlldown ">
      <video
        src="/videos/night.mp4"
        loop
        muted
        autoPlay
        className="background-video" // 백그라운드 비디오 클래스 이름 변경
      />
      <section
        className={`w-2/5 h-full bg-slate-50 flex flex-col justify-center px-16 
          ${isAnimaged ? "animationSlideUp animationPadein" : "opacity-0"}
        `}
      >
        <p className="text-[#1D5673] font-semibold text">About OptiFlow</p>

        <div className={`text-2xl mt-3 font-semibold `}>
          <p>어둠 속에서 빛나는 효율,</p>
          <p>배수지의 수요 예측 및 심야 가동을 통한 비용 절감</p>
        </div>
        <div className="mt-10">
          <p>AI가 예측하고, 효율을 디자인하다</p>
          {/* <p>효율을 디자인하다</p> */}
        </div>
      </section>
      <div className="w-3/5 bg-red-50"></div>
      <span className="arrowSpan"></span>
    </div>
  );
}
