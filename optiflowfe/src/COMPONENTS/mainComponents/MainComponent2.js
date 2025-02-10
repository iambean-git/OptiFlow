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
          <p className={``}>최소한의 비용으로</p>
          <p>효율적인 용수 공급을 위한 시스템을 제공합니다</p>
        </div>
        <div className="mt-10">
          <p>주간 전기 요금 00원</p>
          <p>야간 전기 요금 00원</p>
        </div>
      </section>
      <div className="w-3/5 bg-red-50"></div>
      <span className="arrowSpan"></span>
    </div>
  );
}
