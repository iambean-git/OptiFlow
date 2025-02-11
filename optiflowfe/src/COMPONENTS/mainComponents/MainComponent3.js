import "../../css/videoStyle.css";

import { useState } from "react";
import Modal from "../modal/Modal";


export default function MainComponent3() {
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

      <Modal open={modalOpen} close={closeModal} />
    </div>
  );
}
