import "../css/main.css";

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import Dots from "../components/Dots";

import MainComponent1 from "../components/mainComponents/MainComponent1";
import MainComponent2 from "../components/mainComponents/MainComponent2";
import MainComponent3 from "../components/mainComponents/MainComponent3";
import MainComponent4 from "../components/mainComponents/MainComponent4";

export default function Main() {
  const navigate = useNavigate();

  const DIVIDER_HEIGHT = 5;
  const outerRef = useRef();
  const [currentPage, setCurrentPage] = useState(1);

  const [isVisiblePage2, setIsVisiblePage2] = useState(false);
  const [isVisiblePage3, setIsVisiblePage3] = useState(false);
  const [isVisiblePage4, setIsVisiblePage4] = useState(false);


  //토큰정보
  const token = sessionStorage.getItem("token");

  useEffect(() => {
    if (token) {
      navigate("/dashboard");
      return;
    }

    //스크롤 동작을 감지하는 핸들러
    const wheelHandler = (e) => {
      e.preventDefault();
    
      const { deltaY } = e; // deltaY (양수: 아래 스크롤, 음수: 위 스크롤)
      const { scrollTop } = outerRef.current; // 현재 스크롤 위치
      const pageHeight = window.innerHeight; // 화면 세로 길이
    
      if (deltaY > 0) {
        // 🔽 스크롤 내릴 때
        if (scrollTop >= 0 && scrollTop < pageHeight) {
          // 현재 1페이지
          outerRef.current.scrollTo({
            top: pageHeight + DIVIDER_HEIGHT,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(2);
          setIsVisiblePage2(true);
        } else if (scrollTop >= pageHeight && scrollTop < pageHeight * 2) {
          // 현재 2페이지
          outerRef.current.scrollTo({
            top: pageHeight * 2 + DIVIDER_HEIGHT * 2,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(3);
          setIsVisiblePage3(true);
        } else if (scrollTop >= pageHeight * 2 && scrollTop < pageHeight * 3) {
          // 현재 3페이지
          outerRef.current.scrollTo({
            top: pageHeight * 3 + DIVIDER_HEIGHT * 3,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(4);
          setIsVisiblePage4(true);
        } else {
          // 현재 4페이지 (마지막 페이지)
          outerRef.current.scrollTo({
            top: pageHeight * 3 + DIVIDER_HEIGHT * 3,
            left: 0,
            behavior: "smooth",
          });
        }
      } else {
        // 🔼 스크롤 올릴 때
        if (scrollTop >= 0 && scrollTop < pageHeight) {
          // 현재 1페이지
          outerRef.current.scrollTo({
            top: 0,
            left: 0,
            behavior: "smooth",
          });
        } else if (scrollTop >= pageHeight && scrollTop < pageHeight * 2) {
          // 현재 2페이지
          outerRef.current.scrollTo({
            top: 0,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(1);
        } else if (scrollTop >= pageHeight * 2 && scrollTop < pageHeight * 3) {
          // 현재 3페이지
          outerRef.current.scrollTo({
            top: pageHeight + DIVIDER_HEIGHT,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(2);
        } else {
          // 현재 4페이지
          outerRef.current.scrollTo({
            top: pageHeight * 2 + DIVIDER_HEIGHT * 2,
            left: 0,
            behavior: "smooth",
          });
          setCurrentPage(3);
        }
      }
    };
    

    const outerRefCurrent = outerRef.current;
    outerRefCurrent.addEventListener("wheel", wheelHandler);
    return () => {
      // 컴포넌트가 언마운트될 때 이벤트 핸들러를 제거
      outerRefCurrent.removeEventListener("wheel", wheelHandler);
    };
  }, []);

  return (
    <>
      <button
        className={`absolute px-6 py-2 right-6 top-4 z-10 
                    border-2  bg-white bg-opacity-50 rounded-md text-gray-900
                    ${currentPage==3 || currentPage==4 ? "border-gray-500": "border-white"}`
                  }
        onClick={() => {
          navigate("/login");
        }}
      >
        로그인
      </button>

      <div ref={outerRef} className="outer">
        <Dots currentPage={currentPage} />
        {/* <div className="inner bg-yellow-100">1</div> */}
        <MainComponent1 />
        <div className="divider"></div>
        <MainComponent2 isvisible={isVisiblePage2}/>
        <div className="divider"></div>
        <MainComponent3 isvisible={isVisiblePage3}/>
        <MainComponent4 isvisible={isVisiblePage4}/>
      </div>
    </>
  );
}
