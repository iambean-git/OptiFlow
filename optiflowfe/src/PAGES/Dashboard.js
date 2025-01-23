import NavBar from "../components/NavBar";
import WaterFlow from "../components/waterFlow/WaterFlow";

import { FaRegCalendar } from "react-icons/fa";
import React, { useEffect, useState, useRef } from "react";
import DatePicker from "react-datepicker";
import { ko } from 'date-fns/locale';

import "../css/datepicker.css";
import "react-datepicker/dist/react-datepicker.css";

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
    const year = selectedDate.getFullYear();
    const month = String(selectedDate.getMonth() + 1).padStart(2, "0");
    const day = String(selectedDate.getDate()).padStart(2, "0");
    const hours = String(selectedDate.getHours()).padStart(2, "0");
    const minutes = String(selectedDate.getMinutes()).padStart(2, "0");
    const seconds = String(selectedDate.getSeconds()).padStart(2, "0");
    // setTextDate(`${year}-${month}-${day}T${hours}:${minutes}:${seconds}`);
    setTextDate(`T${hours}:${minutes}:${seconds}`);
  }, [selectedDate]);

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
          <div className="h-full bg-blue-50 relative  ">
            <section className="absolute bottom-0 right-0 !focus-within::border-2 !focus-within:border-black ">
              {/* <div className={`absolute bottom-[37px] left-2 bg-[#f2f2f2] px-1 z-10 text-[10px]
                              ${isFocused ? "text-blue-600" : "text-gray-600"}`}>
                DATE
              </div> */}
              <label className={`flex items-center w-60  rounded-md text-gray-700 bg-white
                                ${isFocused ? "border-2 border-[#6c717c]" : "border-2 border-[#e5e7eb]"}`
              } >
                <DatePicker
                  renderCustomHeader={({
                    date,
                    changeYear,
                    changeMonth,
                    decreaseMonth,
                    increaseMonth,
                    prevMonthButtonDisabled,
                    nextMonthButtonDisabled,
                  }) => (
                    <div className="헤더">
                      <button onClick={decreaseMonth} disabled={prevMonthButtonDisabled}
                        className="mx-2">
                        {"<"}
                      </button>

                      {/* 년도 */}
                      <select
                        value={(date.getFullYear())}
                        onChange={({ target: { value } }) => changeYear(value)}
                        className="w-[60px] h-6 pl-1 rounded-lg focus:outline-none text-black"
                      >
                        {years.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                      <span className="ml-1 mr-3">년</span>

                      {/* 월 */}
                      <select
                        value={months[date.getMonth()]}
                        onChange={({ target: { value } }) =>
                          changeMonth(months.indexOf(value))
                        }
                        className="w-[60px] h-6 pl-1 rounded-lg focus:outline-none text-black"
                      >
                        {months.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                      <span className="ml-1">월</span>
                      <button onClick={increaseMonth} disabled={nextMonthButtonDisabled}
                        className="mx-2">
                        {">"}
                      </button>
                    </div>

                  )}
                  ref={datePickerRef} // ref 추가
                  selected={selectedDate}
                  className="px-2 py-3 flex flex-col bg- justify-center  ml-2 text-sm items-center  focus:outline-none"
                  locale={ko}
                  dateFormat={"yyyy/MM/dd HH:mm"}
                  maxDate={maxDate}
                  onFocus={() => setIsFocused(true)} // 포커스 시 상태 업데이트
                  onBlur={() => setIsFocused(false)} // 블러 시(포커스 해제 시) 상태 해제

                  timeIntervals={15} // 30분 간격
                  showTimeSelect
                  open={isFocused}
                  onChange={date => {
                    // date.setHours(0, 0, 0, 0);
                    setSelectedDate(date);

                    // 날짜와 시간이 선택되었을 때 자동으로 달력 닫기
                    if (date.getHours() !== 0 || date.getMinutes() !== 0) {
                      setIsFocused(false);
                    }
                  }}
                />

                <FaRegCalendar className="ml-6" />
              </label>
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
