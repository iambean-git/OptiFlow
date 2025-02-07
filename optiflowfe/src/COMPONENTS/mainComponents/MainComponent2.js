import "./MainComponent.css";
import { useEffect, useRef, useState } from "react";

export default function MainComponent2() {
  // let windowHeight = window.innerHeight;
  const homeTextRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  // const anmationContent = document.getElementById('animation');

  useEffect(() => {
    const handleScroll = () => {
      if (homeTextRef.current) {
        const topPosition = homeTextRef.current.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;

        if (topPosition < windowHeight) {
          setTimeout(() => {
            setIsVisible(true); // 애니메이션 활성화
          }, 300);

          // 이벤트 리스너 제거 (한 번만 실행)
          window.removeEventListener("scroll", handleScroll);
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    
    // 컴포넌트 언마운트 시 이벤트 제거
    return () => {
      window.removeEventListener("scroll", handleScroll);
      
    };
  }, []);

  useEffect(()=>{
    console.log("💙 isVisible : ",isVisible);
    console.log("💙 isVisible : ",isVisible);
  },[isVisible]);
  return (
    <div className="inner scorlldown ">
      <video
        src="/videos/night.mp4"
        loop
        muted
        autoPlay
        className="background-video" // 백그라운드 비디오 클래스 이름 변경
      />
      <section ref={homeTextRef} className=" w-2/5 h-full bg-slate-50 flex flex-col justify-center px-16">
        <p className="text-[#1D5673] font-semibold text">About OptiFlow</p>

        <div className="text-2xl mt-3 font-semibold animation">
          <p className={`${isVisible? "text-blue-400 " : "text-red-500"}`}>최소한의 비용으로</p>
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
