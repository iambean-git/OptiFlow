import React, { useEffect, useState } from "react";
import NavBar from "../components/NavBar";
import { formatTableDate } from "../utils/dateUtils";
import AdminModal from "../components/modal/AdminModal";
import { toast } from 'react-toastify';
import CustomToast from "../components/ui/CustomToast";

export default function Admin() {
    const server = process.env.REACT_APP_SERVER_ADDR;

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // 에러 상태 저장
    const [inquiriesList, setInquiriesList] = useState([]);
    const [selectedData, setSelectedData] = useState(null);

    const state_new = <span className="border rounded-md px-5 py-1 border-[#e53870] text-[#e53870]">new</span>;
    const state_read = <span className="border rounded-md px-2 py-1 border-[#3a93ee] text-[#3a93ee]">승인 대기</span>;
    const state_approved = <span className="border rounded-md px-2 py-1 border-[#4ba650] text-[#4ba650]">승인 완료</span>;

    const [modalOpen, setModalOpen] = useState(false);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단

        try {
            const url = `${server}/api/inquiries`;
            const response = await fetch(url, {
                signal: controller.signal,
            });

            clearTimeout(timeoutId); // 응답이 오면 타이머 제거

            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

            const data = await response.json();
            // console.log("💌[Admin] 이용문의 확인 : ", data);
            setInquiriesList(data);
        } catch (err) {
            console.error("❌[Admin] fetchData 실패:", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleClickRow = (selectedInquiry) => {
        // console.log(selectedInquiry.inquiryId + "번 문의 클릭");
        setSelectedData(selectedInquiry);
        // 선택된 행이 이미 확인된 경우 변경하지 않음
        if (selectedInquiry.staffConfirmed) return;
        // inquiriesList에서 해당 문의의 staffConfirmed를 true로 변경
        const updatedList = inquiriesList.map((inquiry) =>
            inquiry.inquiryId === selectedInquiry.inquiryId
                ? { ...inquiry, staffConfirmed: true }
                : inquiry
        );
        // 상태 업데이트
        setInquiriesList(updatedList);
        // 서버에 업데이트 요청
        updateInquiryConfirmed(selectedInquiry.inquiryId);
    }

    const updateInquiryConfirmed = async (inquiryId) => {
        try {
            const response = await fetch(`${server}/api/inquiries/confirm/${inquiryId}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                }
            });

            if (!response.ok) throw new Error(`updateInquiryConfirmed 실패! 상태 코드: ${response.status}`);
            // console.log(`✅ 문의 ${inquiryId}번 confirmed 업데이트 완료`);
        } catch (err) {
            console.error("❌ updateInquiryConfirmed 중 오류 발생:", err);
        }
    }

    const updateInquiryApproved = async (inquiryId) => {
        // 이미 승인된 상태면 패스
        if (selectedData.approved) return;
        // 서버에 업데이트 요청
        try {
            const response = await fetch(`${server}/api/inquiries/approve/${inquiryId}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                }
            });
            if (!response.ok) throw new Error(`updateInquiryApproved 실패! 상태 코드: ${response.status}`);
            // console.log(`✅ 문의 ${inquiryId}번 approved 업데이트 완료`);
        } catch (err) {
            console.error("❌ updateInquiryApproved 중 오류 발생:", err);
        }

        // 화면 상태 업데이트
        setInquiriesList((prevList) =>
            prevList.map((inquiry) =>
                inquiry.inquiryId === inquiryId ? { ...inquiry, approved: true } : inquiry
            )
        );

        console.log(`✅ 문의 ${inquiryId} 승인 완료`);
        toast(<CustomToast msg={[`${selectedData.name}님 승인 완료!`]} type={"dark"} />, {
            autoClose: 2000, // 2초 후 자동 닫힘
        });
    }

    return (
        <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
            <NavBar />
            <div className="w-full h-screen pl-[260px] flex flex-col">
                <div className="w-full h-[160px] px-10 flex justify-between">
                    {/* 텍스트 */}
                    <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
                        <h1 className="text-4xl font-medium ">이용 문의 관리</h1>
                        <p className="mt-2">항목을 클릭하여, <span className="whitespace-nowrap"> 세부 정보를 확인 및 승인 처리할 수 있습니다.</span></p>
                    </div>

                </div>
                <section className="px-10 pb-10 pt-6 w-full h-full flex">
                    {/* 테이블 */}
                    <div className="flex-grow pr-4 ">
                        <table className="w-full text-sm text-left rtl:text-right text-gray-500 shadow-md rounded-lg">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th scope="col" className="w-10 py-3 text-center">
                                        번호
                                    </th>
                                    <th scope="col" className="text-center py-3">
                                        이름
                                    </th>
                                    <th scope="col" className="text-center py-3">
                                        지역
                                    </th>
                                    <th scope="col" className="px-6 py-3">
                                        메일
                                    </th>
                                    <th scope="col" className="px-6 py-3">
                                        일시
                                    </th>
                                    <th scope="col" className="px-6 py-3">
                                        문의사항
                                    </th>
                                    <th scope="col" className="py-3">
                                    </th>
                                </tr>
                            </thead>
                            <tbody>

                                {inquiriesList.map((i, idx) => (
                                    <tr
                                        key={i.inquiryId}
                                        className={` border-b border-gray-200  hover:cursor-pointer
                                            ${selectedData && selectedData.inquiryId == i.inquiryId ? " bg-blue-100" : "bg-white hover:bg-gray-50"}
                                            ${i.staffConfirmed ? "" : "font-semibold text-[#333]"}`}
                                        onClick={() => { handleClickRow(i) }}
                                    >
                                        <td className="w-[3%] min-w-10 py-3 text-center">{idx + 1}</td>
                                        <td className="w-[8%] min-w-16 text-center py-4">{i.name}</td>
                                        <td className="w-[10%] min-w-20 text-center py-4">{i.location}</td>
                                        <td className="px-6 py-4">{i.email}</td>
                                        <td className="px-6 py-4">{formatTableDate(i.createdDt)}</td>
                                        <td className="px-6 py-4">
                                            {modalOpen || i.inquiryDetails.length > 42 ? i.inquiryDetails.substr(0, 35) + "..." : i.inquiryDetails}
                                        </td>
                                        <td className="w-[12%] pr-6 text-right py-4">
                                            {i.staffConfirmed ? i.approved ? state_approved : state_read : state_new}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {/* 테이블 종료*/}

                    {/* 문의 상세 보기 */}
                    <div className={`bg-white h-full w-1/4 shadow-md rounded-lg ${selectedData ? "block" : "hidden"}`}>
                        <div className="w-full h-full rounded-lg">
                            <AdminModal data={selectedData} close={() => setSelectedData(null)} updatedApproved={updateInquiryApproved} />
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
