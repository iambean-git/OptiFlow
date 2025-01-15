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
    const hours = cTime.getHours().toString().padStart(2, '0');  // getHours()로 시간만 가져오고 "시"는 제거
    const minutes = cTime.getMinutes().toString().padStart(2, '0');  // 분을 2자리로 포매팅
    const seconds = cTime.getSeconds().toString().padStart(2, '0');  // 초를 2자리로 포매팅

    return (
        <>
            <div className="font-semibold"> 
                {year}년 {month}월 {day}일</div>
    
            <div className="text-4xl font-bold">
                {hours}:{minutes}:{seconds} 
            </div>
        </>

    )
}
