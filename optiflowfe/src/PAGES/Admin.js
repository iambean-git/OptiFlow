import React, { useEffect, useState } from "react";
import NavBar from "../components/NavBar";
import LoadingSpinner from "../components/ui/LoadingSpinner";

export default function Admin() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // 에러 상태 저장
    const [inquiriesList, setInquiriesList] = useState([]);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

        try {
            const url = `http://10.125.121.226:8080/api/inquiries`;
            const response = await fetch(url, {
                signal: controller.signal,
            });

            clearTimeout(timeoutId); // 응답이 오면 타이머 제거

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            console.log("💌[Admin] 이용문의 확인 : ", data);
        } catch (err) {
            console.error("❌ [DashBoard] fetchData1st 실패:", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
            <NavBar />
            <div className="w-full h-screen pl-[260px] flex flex-col">
                <div className="w-full h-[160px] px-10 flex justify-between">
                    {/* 텍스트 */}
                    <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
                        <h1 className="text-4xl ">어드민</h1>
                        <p className="mt-2">각 배수지에 마우스를 올리면, <span className="whitespace-nowrap"> 세부 정보를 확인할 수 있습니다.</span></p>
                    </div>

                </div>
                <section className="px-10 pb-10 pt-6 w-full h-full">
                    <div className="w-full h-full border rounded-lg bg-white ">
                        ㅇㅇㅇㅇ
                    </div>
                </section>
            </div>
        </div>
    );
}
