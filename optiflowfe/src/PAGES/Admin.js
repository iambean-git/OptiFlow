import React, { useEffect, useState } from "react";
import NavBar from "../components/NavBar";
import LoadingSpinner from "../components/ui/LoadingSpinner";
import { formatTableDate } from "../utils/dateUtils";
import InquiryModal from "../components/modal/InquiryModal";
import AdminModal from "../components/modal/AdminModal";

export default function Admin() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // 에러 상태 저장
    const [inquiriesList, setInquiriesList] = useState([]);
    const [trs, setTrs] = useState("");
    const [selectedData, setSelectedData] = useState(null);

    const [modalOpen, setModalOpen] = useState(false);
    // const [modalData, setModalData] = useState('');

    const openModal = () => {
        // console.log("openModal");
        // setModalData(data)
        setModalOpen(true);
    };
    const closeModal = () => {
        setModalOpen(false);
    };


    useEffect(() => {
        console.log("💌[Admin] 렌더링 : ");

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
            setInquiriesList(data);
        } catch (err) {
            console.error("❌ [DashBoard] fetchData1st 실패:", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleClickRow = (selectedInquiry) => {
        console.log(selectedInquiry.inquiryId + "번 문의 클릭");
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
            const response = await fetch(`http://10.125.121.226:8080/api/inquiries/confirm/${inquiryId}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                }
            });

            if (!response.ok) {
                throw new Error(`서버 업데이트 실패! 상태 코드: ${response.status}`);
            }

            console.log(`✅ 문의 ${inquiryId} 상태 업데이트 완료`);
        } catch (err) {
            console.error("❌ 상태 업데이트 중 오류 발생:", err);
        }
    }

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
                <section className="px-10 pb-10 pt-6 w-full h-full flex">
                    {/* 테이블 */}
                    <div className="flex-grow pr-4 ">
                        <table className="w-full text-sm text-left rtl:text-right text-gray-500 shadow-md rounded-lg">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700">
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
                                {/* <tr className="bg-white border-b  border-gray-200 hover:bg-gray-50  font-semibold">
                                    <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap ">
                                        1
                                    </th>
                                    <td className="px-6 py-4">
                                        조은빈
                                    </td>
                                    <td className="px-6 py-4">
                                        부산
                                    </td>
                                    <td className="px-6 py-4">
                                        abcd1234@gmail.com
                                    </td>
                                    <td className="px-6 py-4">
                                        2025-02-12 09:47
                                    </td>
                                    <td className="px-6 py-4">
                                        없습니다.
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className="border rounded-md px-2 py-1">new</span>
                                    </td>
                                </tr> */}
                                {/* {trs} */}
                                {inquiriesList.map((i, idx) => (
                                    <tr
                                        key={i.inquiryId}
                                        className={` border-b border-gray-200  hover:cursor-pointer
                                            ${selectedData && selectedData.inquiryId == i.inquiryId ? " bg-blue-100" : "bg-white hover:bg-gray-50"}
                                            ${i.staffConfirmed ? "" : "font-semibold"}`}
                                        onClick={() => { handleClickRow(i) }}
                                    >
                                        <td className="w-[3%] min-w-10 py-3 text-center">{idx + 1}</td>
                                        <td className="w-[8%] min-w-16 text-center py-4">{i.name}</td>
                                        <td className="w-[10%] min-w-20 text-center py-4">{i.location}</td>
                                        <td className="px-6 py-4">{i.email}</td>
                                        <td className="px-6 py-4">{formatTableDate(i.createdDt)}</td>
                                        <td className="px-6 py-4">{i.inquiryDetails}</td>
                                        <td className="w-[8%] text-center py-4">
                                            {i.staffConfirmed ? "" : <span className="border rounded-md px-2 py-1">new</span>}
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
                            <AdminModal data={selectedData} close={() => setSelectedData(null)} />
                        </div>
                    </div>
                </section>
            </div>
            <InquiryModal open={modalOpen} close={closeModal} />
        </div>
    );
}
