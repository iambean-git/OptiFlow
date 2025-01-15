import "../CSS/main.css";

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import Dots from "../COMPONENTS/Dots";

import MainComponent1 from "../COMPONENTS/mainComponents/MainComponent1";
import MainComponent2 from "../COMPONENTS/mainComponents/MainComponent2";
import MainComponent3 from "../COMPONENTS/mainComponents/MainComponent3";;

export default function Main() {
  const navigate = useNavigate();

  const DIVIDER_HEIGHT = 5;
  const outerRef = useRef();
  const [currentPage, setCurrentPage] = useState(1);

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

      const { deltaY } = e;   // deltaY (양수:아래스크롤/음수:위스크롤)
      const { scrollTop } = outerRef.current; // 스크롤 위쪽 끝부분 위치
      const pageHeight = window.innerHeight; // 화면 세로길이

      if (deltaY > 0) {
        // 스크롤 내릴 때
        if (scrollTop >= 0 && scrollTop < pageHeight) {
          //현재 1페이지
          // console.log("현재 1페이지, down");
          outerRef.current.scrollTo({
            top: pageHeight + DIVIDER_HEIGHT, left: 0, behavior: "smooth",
          });
          setCurrentPage(2);
        } else if (scrollTop >= pageHeight && scrollTop < pageHeight * 2) {
          //현재 2페이지
          // console.log("현재 2페이지, down");
          outerRef.current.scrollTo({
            top: pageHeight * 2 + DIVIDER_HEIGHT * 2, left: 0, behavior: "smooth",
          });
          setCurrentPage(3);
        } else {
          // 현재 3페이지
          // console.log("현재 3페이지, down");
          outerRef.current.scrollTo({
            top: pageHeight * 2 + DIVIDER_HEIGHT * 2, left: 0, behavior: "smooth",
          });
        }
      } else {
        // 스크롤 올릴 때
        if (scrollTop >= 0 && scrollTop < pageHeight) {
          //현재 1페이지
          // console.log("현재 1페이지, up");
          outerRef.current.scrollTo({
            top: 0, left: 0, behavior: "smooth",
          });
        } else if (scrollTop >= pageHeight && scrollTop < pageHeight * 2) {
          //현재 2페이지
          // console.log("현재 2페이지, up");
          outerRef.current.scrollTo({
            top: 0, left: 0, behavior: "smooth",
          });
          setCurrentPage(1);
        } else {
          // 현재 3페이지
          // console.log("현재 3페이지, up");
          outerRef.current.scrollTo({
            top: pageHeight + DIVIDER_HEIGHT, left: 0, behavior: "smooth",
          });
          setCurrentPage(2);
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

      <button className="absolute border border-gray-400 rounded-md px-6 py-2 right-6 top-4 z-10"
        onClick={() => { navigate("/login") }}>
        로그인
      </button>

      <div ref={outerRef} className="outer">
        <Dots currentPage={currentPage} />
        {/* <div className="inner bg-yellow-100">1</div> */}
        <MainComponent1 />
        <div className="divider"></div>
        <MainComponent2 />
        <div className="divider"></div>
        <MainComponent3 />
      </div>
    </>
  )
}
