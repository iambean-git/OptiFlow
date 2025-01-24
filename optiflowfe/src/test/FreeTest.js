import { FaRegCalendar } from "react-icons/fa";
import React, { useState } from "react";
import DatePicker from "react-datepicker";
import { ko } from 'date-fns/locale';

import "react-datepicker/dist/react-datepicker.css";
import CustomDatePicker from "../components/datePicker/CustomDatePicker";

export default function FreeTest() {
    const [startDate, setStartDate] = useState(new Date() - 1);
    return (
        <>
            {/* <label className="flex items-center w-60 bg-[#f4f4f4] rounded-md text-gray-700" >  */}
                <DatePicker selected={startDate}
                    className="p-2  flex justify-center  ml-2 text-center items-center bg-[#f4f4f4] focus:border-0"
                    locale={ko}
                    dateFormat={"yyyy년 MM월 dd일"}
                    maxDate={new Date()}
                    custo
                    onChange={date => setStartDate(date)} 
                    shouldCloseOnSelect = {true}
                    onClickOutside={() => {}}
                    // shouldCloseOnSelect={true} // 날짜 선택 시 자동으로 닫힘
                />
                
                <FaRegCalendar className="ml-3"/>

                <CustomDatePicker/>
            {/* </label> */}
        </>
    );
}
