import NavBar from "../components/NavBar";
import WaterFlow from "../components/waterFlow/WaterFlow";

import { FaRegCalendar } from "react-icons/fa";
import React, { useEffect, useState, useRef, forwardRef } from "react";
import DatePicker from "react-datepicker";
import { ko } from 'date-fns/locale';

// import "../css/datepicker.css";
import "react-datepicker/dist/react-datepicker.css";
import DateNTime from "../components/datepicker/DateNTime";

export default function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [maxDate, setMaxDate] = useState(new Date());
  const [isFocused, setIsFocused] = useState(false); // 포커스 상태 관리
  const [textDate, setTextDate] = useState("");
  const datePickerRef = useRef(null); // DatePicker의 ref 

  function range(start, end, step = 1) {
    const result = [];
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
    return result;
  }

  const years = range(2023, new Date().getFullYear() + 1, 1);

  const months = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
  ];

  // 
  useEffect(() => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1); // 하루 전으로 설정
    yesterday.setHours(10, 0, 0, 0); // 오전 10시로 설정 (분, 초, 밀리초는 0으로 초기화)
    setSelectedDate(yesterday);

    const maxdate = new Date();
    maxdate.setDate(maxdate.getDate() - 1); // 하루 전으로 설정
    maxdate.setHours(23, 59, 59, 999); // 어제의 끝으로 설정
    setMaxDate(maxdate);
  }, []);

  useEffect(() => {
    console.log("[Dashboard] 날짜 및 시간 선택 : ", selectedDate);
    const year = selectedDate.getFullYear();
    const month = String(selectedDate.getMonth() + 1).padStart(2, "0");
    const day = String(selectedDate.getDate()).padStart(2, "0");
    const hours = String(selectedDate.getHours()).padStart(2, "0");
    const minutes = String(selectedDate.getMinutes()).padStart(2, "0");
    const seconds = String(selectedDate.getSeconds()).padStart(2, "0");
    // setTextDate(`${year}-${month}-${day}T${hours}:${minutes}:${seconds}`);

    setTextDate(`T${hours}:${minutes}:${seconds}`);
  }, [selectedDate]);

  const CustomInput = forwardRef(
    ({ value, onClick }, ref) => (
      <div className="mx-2 px-12 py-2 flex items-center relative bg-white border rounded-lg"
        style={{boxShadow:"0px 0px 15px rgba(0, 0, 0, 0.15)"}}
        onClick={onClick} ref={ref}>
        <FaRegCalendar className='absolute left-4' />
        <p className='left-2'>{value}</p>
      </div>
    ),
  );
  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <div className="w-full h-[160px] px-10 flex justify-between">
          {/* 텍스트 */}
          <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
            <h1 className="text-4xl ">타이틀</h1>
            <p className="mt-2">각 배수지를 클릭하면, <span className="whitespace-nowrap"> 세부 정보를 확인할 수 있습니다.</span></p>
          </div>

          {/* 달력 */}
          <div className="h-full  relative min-w-72 ">
            <section className="absolute bottom-0 right-0 ">
              <DateNTime />
            </section>

          </div>
        </div>
        <section className="px-10 pb-10 pt-6 w-full h-full">
          <div className="w-full h-full border rounded-lg ">
            <WaterFlow selectedDate={textDate} />
          </div>
        </section>
      </div>
    </div>
  );
}
