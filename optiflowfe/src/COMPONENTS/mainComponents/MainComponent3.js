import "../../css/videoStyle.css";
import { AiOutlineDashboard } from "react-icons/ai";
import { SlGraph } from "react-icons/sl";
import { BsGraphUp } from "react-icons/bs";
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
      toast(<CustomToast msg={["이용 문의 접수가 완료되었습니다.", "승인 완료시, 이메일을 통해 확인하실 수 있습니다."]} />, {
        // autoClose: 3000, // 3초 후 자동 닫힘
        progressStyle: { backgroundColor: "#4caf50" }, // 초록색으로 변경 
        progressClassName: " w-[300px]", // 프로그레스 바 색상 및 높이 조정
      });
    }
  };

  return (
    <div className="main-component-container flex justify-center">

      <video
        // src="/videos/수돗물.mp4"
        src="/videos/waterdrop2.mp4"
        loop
        muted
        autoPlay
        className="background-video opacity-30"
      />

      <div className="w-[55%] h-full grid grid-rows-2 py-20 p-4 gap-10">

        {/* ==========  1번째 영역 [DASHBOARD] ========== */}
        <section className="flex justify-between">
          <div className="flex items-center justify-center h-full rounded-lg  ">
            <img src="/images/mainImg/cap_dashboard1.png" alt="dashboard"
              className="rounded-lg mr-2 h-[75%] shadow-lg" />
            <img src="/images/mainImg/cap_dashboard2.png" alt="dashboard"
              className="rounded-lg h-4/5  " />
          </div>

          <div className="flex flex-col justify-center mr-20">
            <p className="text-lg text-blue-500 font-semibold flex items-center">
              <AiOutlineDashboard className="mr-1" />
              DASHBOARD
            </p>
            <p className="text-3xl font-bold">
              실시간 배수지 정보를 한눈에
            </p>
            <p className="text-lg">

            </p>
            <div className="text-lg text-gray-500 mt-6">
              <p className="">
                전체 배수지의 실시간 저수량 모니터링
              </p>
              <p className="">
                각 배수지의 지난 24시간 유입량 및 유출량 확인
              </p>
              <p className="">
                미래 24시간 예상 유출량 및 최적 유입량
              </p>
            </div>


          </div>
        </section>

        {/* ==========  2번째 영역 [visualization ] ========== */}
        <section className="flex justify-between items-center">
          <div className="flex flex-col justify-center ml-14">
            <p className="text-lg text-blue-500 font-semibold flex items-center">
              <SlGraph className="mr-1" />
              VISUALIZATION
            </p>
            <p className="text-3xl font-bold">
              기계학습 기반 물 소비량 예측
            </p>
            <p className="text-lg">

            </p>
            <div className="text-lg text-gray-500 mt-6">
              <p className="">
              AI 모델을 활용해 시간당 유출량을 예측하고,
              </p>
              <p className="">
              전기 요금 절약을 위한 개선된 운영 계획을 제공합니다.
              </p>
              {/* <p className="">
                미래 24시간 예상 유출량 및 최적 유입량
              </p> */}
            </div>


          </div>
          <img src="/images/mainImg/cap_graph.png" alt="dashboard"
            className="rounded-lg ml-12 h-4/5 shadow-xl" />
        </section>


      </div>
      {/* <div className="w-3/5 h-full">
        <button className="px-4 py-1 border rounded-md" onClick={openModal}>
          이용 문의하기
        </button>
      </div> */}

      <InquiryModal open={modalOpen} close={closeModal} />
      {/* <ToastContainer /> */}
    </div>
  );
}
