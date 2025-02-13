import "./modal.css";
import React, { useState } from "react";
import AnimatedInput from "../ui/AnimatedInput";
import AnimatedTextarea from "../ui/AnimatedTextArea";

export default function InquiryModal({ open, close, data, isAdmin = false }) {
    const [name, setName] = useState("");
    const [contact, setContact] = useState("");
    const [email, setEmail] = useState("");
    const [tmpid, setTmpid] = useState("");
    const [location, setLocation] = useState("");
    const [inquiry, setInquiry] = useState("");

    const handleClose = () => {
        setName("");
        setContact("");
        setEmail("");
        setTmpid("");
        setLocation("");
        setInquiry("");
        close();
    }

    const handleSubmit = () => {
        fetchPost();
    }

    const fetchPost = async () => {
        try {
            const url = `http://10.125.121.226:8080/api/inquiries`;
            const postData = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    contact: contact,
                    email: email,
                    location: location,
                    tmpid: tmpid,
                    inquiryDetails: inquiry
                })
            };
            const resp = await fetch(url, postData);
            if (!resp.ok) throw new Error(`HTTP error! Status: ${resp.status}`);
            handleClose();
            close(true);

        } catch (error) {
            console.error("❌ [Modal] fetchPost 실패:", error);
            close(false);
        }
    };

    return (
        <div className={open ? 'openModal modal' : 'modal'}>
            {open ? (
                // 모달이 열릴때 openModal 클래스가 생성된다.
                <section className='flex flex-col '>

                    <main className='w-full flex-grow p-4 flex flex-col items-center justify-start relative'>
                        <button className="absolute right-4 text-right text-[#999] text-xl" onClick={handleClose}>
                            &times;
                        </button>
                        {/* 제목 */}
                        <div className='w-[450px] h-16 flex justify-center items-center text-2xl font-bold text-[#3b82f6] '>
                            OPTIFLOW 이용 문의
                        </div>
                        <div id="map" className='h-5/6 w-[450px] flex flex-col justify-center items-center'>
                            <AnimatedInput
                                type={"text"} label={"이름"} value={name}
                                onChange={(e) => setName(e.target.value)}
                                required={true}
                            />
                            <AnimatedInput
                                type={"text"} label={"연락처"} value={contact}
                                onChange={(e) => setContact(e.target.value)}
                            />
                            <AnimatedInput
                                type={"text"} label={"메일"} value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required={true}
                            />
                            <AnimatedInput
                                type={"text"} label={"이용 희망 지역"} value={location}
                                onChange={(e) => setLocation(e.target.value)}
                                detail={"서비스를 이용하실 지역을 입력해주세요."}
                            />
                            <AnimatedInput
                                type={"text"} label={"희망 아이디"} value={tmpid}
                                onChange={(e) => setTmpid(e.target.value)}
                                detail={"이용 승인시, 희망하는 아이디를 입력해주세요."}
                                required={true}
                            />
                            <AnimatedTextarea
                                label={"기타 문의사항"} value={inquiry}
                                onChange={(e) => setInquiry(e.target.value)}
                                detail="추가적인 정보를 입력하세요."
                                rows={3}
                            />

                        </div>

                    </main>
                    <footer className="pb-3 px-4 mb-6 text-center">
                        <button className="w-[300px] py-2 bg-[#3b82f6] rounded-md text-white
                                            disabled:cursor-not-allowed disabled:opacity-45"
                            disabled={!name || !email || !tmpid}
                            onClick={handleSubmit}>
                            제 출
                        </button>
                    </footer>
                </section>
            ) : null}
        </div>
    )
}
