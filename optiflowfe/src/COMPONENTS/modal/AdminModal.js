import React from 'react';
import emailjs from 'emailjs-com';

export default function AdminModal({ data, close }) {
    const handleAdmitClick = () => {

        const id = data.tmpid;
        const pwd = "opti" + generateRandomNum();

        console.log("[승인]");
        console.log("[id] : ", id);
        console.log("[pw] : ", pwd);

        const fetchParams = {
            username: id,
            password: pwd
        };

        //이메일 내용
        const emailParams = {
            to_email: data.email,
            userID: id,
            userPWD: pwd
        };

        // emailjs.send(
        //     'optiflow',  //service id
        //     'template_approved', // 템플릿 ID
        //     emailParams,
        //     process.env.REACT_APP_EMAILJS_KEY
        // ).then((response) => {
        //     console.log('이메일이 성공적으로 보내졌습니다:', response);
        //     fetchMakeNewMember(fetchParams);
        //     // setIsEmailSent(true);
        //     // 이메일 전송 성공 처리 로직 추가
        // }).catch((error) => {
        //     console.error('이메일 보내기 실패:', error);
        //     // 이메일 전송 실패 처리 로직 추가
        // });

        fetchMakeNewMember(fetchParams);
    };

    const fetchMakeNewMember = async(fetechData) => {
        try {
            // const url = `http://10.125.121.226:8080/api/inquiries`;
            // const postData = {
            //     method: 'POST',
            //     headers: {
            //         'Content-Type': 'application/json'
            //     },
            //     body: JSON.stringify(fetechData)
            // };
            // const resp = await fetch(url, postData);
            // if (!resp.ok) throw new Error(`HTTP error! Status: ${resp.status}`);

            // console.log("가입 성공 resp:", resp);
            // 성공했다고 먼가 띄우고싶음
            close();
            


        } catch (error) {
            console.error("❌ [Modal] fetchMakeNewMember 실패:", error);
        }

        window.location.reload();
        // alert("ddd");
    }

    const generateRandomNum = () => {
        return Math.floor(1000 + Math.random() * 9000).toString();
    };

    return (
        <div className='w-full h-full rounded-lg relative'>
            {!data ? <></> :
                <>
                    <button className="absolute top-4 right-6 text-right text-[#999] text-xl z-10" onClick={close}>
                        &times;
                    </button>

                    <main className='w-full flex-grow p-4 flex flex-col items-center justify-start relative'>

                        <div className='h-16 flex justify-center items-center text-xl font-bold text-[#3b82f6] '>
                            OPTIFLOW 이용 문의
                        </div>
                        <div className=' w-full flex flex-col justify-center items-center'>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>이름</label>
                                <div className='w-full h-10 px-3 py-2 border border-gray-300 rounded-md mb-4'>{data.name}</div>
                            </div>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>연락처</label>
                                <div className='w-full h-10 px-3 py-2 border border-gray-300 rounded-md mb-4'>{data.contact}</div>
                            </div>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>메일</label>
                                <div className='w-full h-10 px-3 py-2 border border-gray-300 rounded-md mb-4'>{data.email}</div>
                            </div>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>이용 희망 지역</label>
                                <div className='w-full h-10 px-3 py-2 border border-gray-300 rounded-md mb-4'>{data.location}</div>
                            </div>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>희망 아이디</label>
                                <div className='w-full h-10 px-3 py-2 border border-gray-300 rounded-md mb-4'>{data.tmpid}</div>
                            </div>
                            <div className='w-11/12'>
                                <label className='text-gray-600 text-sm'>기타 문의사항</label>
                                <div className='w-full h-24 px-3 py-2 border border-gray-300 rounded-md mb-4 overflow-y-auto'>{data.inquiryDetails}</div>
                            </div>
                        </div>
                    </main>
                    <footer className="w-full pb-3 px-4 mb-6 text-center">
                        <button className="w-11/12 py-2 bg-[#3b82f6] rounded-md text-white" onClick={handleAdmitClick}>
                            승인하기
                        </button>
                    </footer>
                </>
            }
        </div>
    )
}
