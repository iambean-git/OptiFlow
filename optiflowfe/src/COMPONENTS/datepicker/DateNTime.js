import React, { useState, useEffect, forwardRef } from "react";
import { FaRegCalendar } from "react-icons/fa";
import { FaRegClock } from "react-icons/fa";
import DatePicker from "react-datepicker";
import { ko } from 'date-fns/locale';
import "react-datepicker/dist/react-datepicker.css";
import "./datepickerwithoption.css"

import { MaxDate } from "../../recoil/DateAtom";
import { useRecoilValue } from "recoil";

export default function DateNTime({ selectedDate, setSelectedDate }) {
    const maxDate = useRecoilValue(MaxDate);

    function range(start, end, step = 1) {
        const result = [];
        for (let i = start; i < end; i += step) {
            result.push(i);
        }
        return result;
    }

    const years = range(2023, maxDate.getFullYear()+1, 1);

    const months = [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    ];

    const CustomInput = forwardRef(
        ({ value, onClick }, ref) => (
            <div className="bg-white w-[200px] h-[40px] rounded-md flex items-center relative justify-center" onClick={onClick} ref={ref}>
                <FaRegCalendar className="absolute left-2 text-[#7c8bdb]" />
                <span className="pl-2">
                    {value} {/* 기본 텍스트 설정 */}
                </span>
            </div>
        ),
    );

    const CustomTimeInput = forwardRef(
        ({ value, onClick }, ref) => (
            <div className="bg-white w-[140px] h-[40px] rounded-md flex items-center relative justify-center" onClick={onClick} ref={ref}>
                <FaRegClock className="absolute left-2 text-[#7c8bdb]" />
                <span className="pl-2">
                    {value} {/* 기본 텍스트 설정 */}
                </span>
            </div>
        ),
    );

    return (
        <div className="flex">
            <div className="bg-gray-50 p-0.5 w-52 h-12 rounded-md border border-[#7c8bdb] flex items-center">
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
                            <button onClick={increaseMonth} disabled={nextMonthButtonDisabled}
                                className="mx-2 disabled:cursor-not-allowed movingArrow">
                                {">"}
                            </button>
                        </div>

                    )}
                    selected={selectedDate}
                    // className="px-2 py-3 flex flex-col bg- justify-center  ml-2 text-sm items-center  focus:outline-none"
                    locale={ko}
                    dateFormat={"yyyy년 MM월 dd일"}
                    maxDate={maxDate}
                    timeIntervals={60} // 30분 간격
                    onChange={date => setSelectedDate(date)}
                    customInput={<CustomInput />}
                    showPopperArrow={false}
                />
            </div>

            <div className="ml-2 bg-gray-50 p-0.5 w-36 h-12 rounded-md border border-[#7c8bdb] flex items-center">
                <DatePicker
                    selected={selectedDate}
                    // className="px-2 py-3 flex flex-col bg- justify-center  ml-2 text-sm items-center  focus:outline-none"
                    locale={ko}
                    dateFormat={"HH시 mm분"}
                    maxDate={maxDate}
                    timeIntervals={60} // 30분 간격
                    showTimeSelectOnly={true}
                    showTimeSelect
                    onChange={date => setSelectedDate(date)}
                    customInput={<CustomTimeInput />}
                    showPopperArrow={false}
                />
            </div>
        </div>

    )
}
