import { useState, useEffect } from "react";

export default function LocalTime() {
    const [cTime, setCtime] = useState(new Date());

    useEffect(() => {
        const tm = setInterval(() => {
            setCtime(new Date());
        }, 1000);   //1000ms 즉,1초에 한번 함수 실행

        return () => { clearInterval(tm) };   //컴포넌트가 없어질 때 클리어(setInterval 종료)됨
    }, []);

    // 날짜 ex)2025년 01월 14일
    const year = cTime.getFullYear();
    const month = (cTime.getMonth() + 1).toString().padStart(2, '0');  // 월은 0부터 시작하므로 1을 더해줌
    const day = cTime.getDate().toString().padStart(2, '0');

    //시간
    const hours = cTime.toLocaleString('en-US', { hour: '2-digit', hour12: false });
    const minutes = cTime.toLocaleString('en-US', { minute: '2-digit' });

    return (
        <>
            <div className="font-semibold"> 
                {year}년 {month}월 {day}일</div>
            <div className="text-4xl font-bold">
                {hours} : {minutes}
            </div>
        </>

    )
}
