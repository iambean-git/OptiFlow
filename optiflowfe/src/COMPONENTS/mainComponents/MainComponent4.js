import "../../css/videoStyle.css";
import { AiOutlineDashboard } from "react-icons/ai";
import { BsDiagram3 } from "react-icons/bs";
import { useState } from "react";
import InquiryModal from "../modal/InquiryModal";
import { ToastContainer, toast } from 'react-toastify';
import CustomToast from "../ui/CustomToast";

export default function MainComponent4() {
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

      {/* <video
        // src="/videos/수돗물.mp4"
        src="/videos/waterdrop2.mp4"
        loop
        muted
        autoPlay
        className="background-video opacity-30"
      /> */}

      <div className="w-full h-full flex flex-col  justify-center items-center">



        {/* ==========  3번째 영역 [] ========== */}
        <section className="w-[55%] grow flex justify-between items-center px-14">
          <img src="/images/mainImg/cap_waterflow.png" alt="dashboard"
            className="rounded-lg mr-12 h-1/2 shadow-xl border" />
          {/* <img src="/images/mainImg/cap_dashboard2.png" alt="dashboard"
            className="rounded-lg mr-4" /> */}
          <div className="flex flex-col justify-center">
            <p className="text-lg text-blue-500 font-semibold flex items-center">
              <BsDiagram3 className="ml-0.5 mr-1.5" />
              FLOW
            </p>
            <p className="text-3xl font-bold">
            배수지 단위 데이터 저장 및 관리
            </p>
            <p className="text-lg">

            </p>
            <div className="text-lg text-gray-500 mt-6">
              <p className="">
              배수지별 상세 정보를 구분 관리하고,
              </p>
              <p className="">
              과거 수위 기록에 대한 시각적 모식도를 제공하여, 
              </p>
              <p className="">
              직관적으로 데이터를 확인할 수 있도록 합니다.
              </p>
            </div>


          </div>
        </section>
        
        <div className="w-full h-1/5 bg-[#eef2f9] flex justify-center ">
          <div className="w-[55%] h-full flex flex-col justify-center items-center">
            <p className="text-xl mb-5 font-semibold">
              OPTIFLOW와 함께 하세요.
            </p>
            <button className="px-12 py-1.5 border-2 border-[#0d4296] text-[#0d4296] rounded-md
                                hover:bg-white transition-all"
              onClick={openModal}>
              이용 문의하기
            </button>
          </div>
        </div>

        <footer className="w-full h-1/5 flex justify-center bg-[#293a55]">
          <div className="w-3/5 h-full flex justify-between items-center">
            <div className=" text-gray-400 ">
              <p className="text-lg font-bold mb-2">
                OPTIFLOW
              </p>
              <p className="text-sm my-0.5">
                <span className="mr-2">만든이</span> 윤찬희 정원영 조은빈
              </p>
              <p className="text-sm my-0.5">
                PNU K-Digital Training 8
              </p>
              <p className="text-sm my-0.5">
                (46241) 부산광역시 금정구 부산대학로63번길 2 (장전동) 부산대학교
              </p>
            </div>
            <img src="/images/logo_square_white.png" alt="logo"
              className="h-[80%] mt-3" />
          </div>
        </footer>
      </div>


      <InquiryModal open={modalOpen} close={closeModal} />
      {/* <ToastContainer /> */}
    </div>
  );
}
