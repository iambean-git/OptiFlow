import "../../css/videoStyle.css";

import { useState } from "react";
import InquiryModal from "../modal/InquiryModal";
import { ToastContainer, toast } from 'react-toastify';
import CustomToast from "../ui/CustomToast";

export default function MainComponent3() {
  const [modalOpen, setModalOpen] = useState(false);
  // const [modalData, setModalData] = useState('');

  const notify = () => toast("Wow so easy!");

  const openModal = () => {
    // console.log("openModal");
    // setModalData(data)
    setModalOpen(true);
  };
  const closeModal = (isPosted = false) => {
    setModalOpen(false);
    if (isPosted) {
      // toast.success("이용 문의 접수가 완료되었습니다. 승인 완료시, 이메일을 통해 확인하실 수 있습니다.",{
      //   position: "bottom-center",
      //   // hideProgressBar: false,
      // });
      // toast(<CustomToast msg="🎉 Tailwind 토스트 메시지!" />, { autoClose: false });
      toast(<CustomToast msg={["이용 문의 접수가 완료되었습니다.","승인 완료시, 이메일을 통해 확인하실 수 있습니다."]} />, {
        // autoClose: 3000, // 3초 후 자동 닫힘
        progressStyle: { backgroundColor: "#4caf50" }, // 초록색으로 변경 
        progressClassName: " w-[300px]", // 프로그레스 바 색상 및 높이 조정
      });
    }
  };

  return (
    <div className="inner">

      <div className="w-2/5 h-full relative overflow-hidden"> {/* 부모 div에 relative 적용 */}
        <video
          // src="/videos/수돗물.mp4"
          src="/videos/waterdrop2.mp4"
          loop
          muted
          autoPlay
          className="absolute top-0 left-0 w-full h-full object-cover scale-x-[-1] z-[-1] scale-y-[1.2] object-[50%]"
        />
      </div>


      <div className="w-3/5 h-full">
        <button className="px-4 py-1 border rounded-md" onClick={openModal}>
          이용 문의하기
        </button>
      </div>

      <InquiryModal open={modalOpen} close={closeModal} />
      {/* <ToastContainer /> */}
    </div>
  );
}
