import React, { useState, useEffect, forwardRef } from "react";
import { FaRegCalendar } from "react-icons/fa";

import DatePicker from "react-datepicker";
import { ko } from 'date-fns/locale';
import "react-datepicker/dist/react-datepicker.css";
import "./datepickerwithoption.css"
export default function DatePickerWithOption() {
    const [option, setOption] = useState("day"); // "day", "month", "time"
    const [selectedDate, setSelectedDate] = useState(null);
    const [maxDate, setMaxDate] = useState(new Date());

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

    const ExampleCustomInput = forwardRef(
        ({ value, onClick}, ref) => (
            <div className="bg-white w-[200px] h-[40px] rounded-md flex items-center relative justify-center" onClick={onClick} ref={ref}>
                <FaRegCalendar className="absolute left-2 text-[#7c8bdb]" />
                <span className="pl-2">
                    {value || (option === "time" ? "날짜를 선택하세요" : (option === "month" ? "연도를 선택하세요" : "월을 선택하세요"))} {/* 기본 텍스트 설정 */}
                </span>
            </div>
        ),
    );

    const handleOptionChange = (ops) => {
        setOption(ops);
        setSelectedDate(null); // 옵션 변경 시 선택된 날짜 초기화
    };


    useEffect(() => {
        // const yesterday = new Date();
        // yesterday.setDate(yesterday.getDate() - 1); // 하루 전으로 설정
        // yesterday.setHours(10, 0, 0, 0); // 오전 10시로 설정 (분, 초, 밀리초는 0으로 초기화)
        // setSelectedDate(yesterday);

        const maxdate = new Date();
        maxdate.setDate(maxdate.getDate() - 1); // 하루 전으로 설정
        maxdate.setHours(23, 59, 59, 999); // 어제의 끝으로 설정
        setMaxDate(maxdate);
    }, []);

    // 선택된거 확인
    useEffect(() => {
        if (!selectedDate) return;
        const year = selectedDate.getFullYear();
        const month = String(selectedDate.getMonth() + 1).padStart(2, "0");
        const day = String(selectedDate.getDate()).padStart(2, "0");
        // const hours = String(selectedDate.getHours()).padStart(2, "0");
        // const minutes = String(selectedDate.getMinutes()).padStart(2, "0");
        // const seconds = String(selectedDate.getSeconds()).padStart(2, "0");

        const selected = {
            option: option,
            selectedValue: (
                option === "time" ? `${year}-${month}-${day}` :
                    option === "month" ? `${year}` :
                        `${year}-${month}`
            )
        };
        console.log("selected :", selected);
    }, [selectedDate]);

    return (
        <div className="flex items-center">
            <section className="flex h-12 bg-gray-100 p-0.5 w-fit border border-[#7c8bdb] rounded-md hover:cursor-pointer">
                <div className={`p-2 rounded-tl-md rounded-bl-md border-r w-16 text-center
                                ${option === 'time' ? "bg-[#7c8bdb] text-white" : "bg-white text-black"}`}
                    onClick={() => handleOptionChange("time")}
                >
                    시간별
                </div>
                <div className={`p-2 w-16 text-center 
                                ${option === 'day' ? "bg-[#7c8bdb] text-white" : "bg-white text-black"}`}
                    onClick={() => handleOptionChange("day")}
                >
                    일별
                </div>
                <div className={`p-2 rounded-tr-md rounded-br-md border-l w-16 text-center
                                ${option === 'month' ? "bg-[#7c8bdb] text-white" : "bg-white text-black"}`}
                    onClick={() => handleOptionChange("month")}
                >
                    월별
                </div>
            </section>
            <div className="ml-2 bg-gray-50 p-0.5 w-52 h-12 rounded-md border border-[#7c8bdb]">
                <DatePicker
                    renderCustomHeader={
                        option === "time" ?
                            ({
                                date,
                                changeYear,
                                changeMonth,
                                decreaseMonth,
                                increaseMonth,
                                prevMonthButtonDisabled,
                                nextMonthButtonDisabled,
                            }) => (
                                <div>
                                    <button onClick={decreaseMonth} disabled={prevMonthButtonDisabled}
                                        className="mx-2 movingArrow">
                                        {"<"}
                                    </button>

                                    {/* 년도 */}
                                    <select
                                        value={(date.getFullYear())}
                                        onChange={({ target: { value } }) => changeYear(value)}
                                        className="w-[75px] h-6 pl-1 rounded-sm focus:outline-none customheaderinput"
                                    >
                                        {years.map((option) => (
                                            <option key={option} value={option} className="text-black">
                                                {option}년
                                            </option>
                                        ))}
                                    </select>
                                    {/* <span className="ml-1 mr-3">년</span> */}

                                    {/* 월 */}
                                    <select
                                        value={months[date.getMonth()]}
                                        onChange={({ target: { value } }) =>
                                            changeMonth(months.indexOf(value))
                                        }
                                        className="w-[75px] h-6 pl-1 ml-2 rounded-sm focus:outline-none customheaderinput"
                                    >
                                        {months.map((option) => (
                                            <option key={option} value={option} className="text-black">
                                                {option}월
                                            </option>
                                        ))}
                                    </select>
                                    {/* <span className="ml-1">월</span> */}
                                    <button onClick={increaseMonth} disabled={nextMonthButtonDisabled}
                                        className="mx-2 disabled:cursor-not-allowed movingArrow">
                                        {">"}
                                    </button>
                                </div>

                            ) : undefined
                    }
                    selected={selectedDate}
                    onChange={(date) => setSelectedDate(date)}
                    customInput={<ExampleCustomInput />}
                    dateFormat={
                        option === "day"
                            ? "yyyy년 MM월"
                            : option === "month"
                                ? "yyyy년"
                                : "yyyy년 MM월 dd일"
                    }
                    showYearPicker={option === "month"}
                    showMonthYearPicker={option === "day"}
                    timeIntervals={60} // 시간별 선택 시 15분 단위로 설정
                    locale={ko}
                    maxDate={maxDate}
                    minDate={new Date("2023-01-01")}
                    showPopperArrow={false}

                />
            </div>
        </div>

    );
}