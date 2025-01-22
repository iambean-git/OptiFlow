import React, { useState } from "react";
import DatePicker from "react-datepicker";
// import { ko } from 'date-fns/esm/locale';

import "react-datepicker/dist/react-datepicker.css";

export default function FreeTest() {
    const [startDate, setStartDate] = useState(new Date());
    return (
        <DatePicker selected={startDate} 
        // locale={ko}
        onChange={date => setStartDate(date)} />
    );
}
