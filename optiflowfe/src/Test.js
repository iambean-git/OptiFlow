import "./CSS/test.css";

import { useNavigate } from "react-router-dom";
import WaterDrop from "./test/WaterDropTest";

export default function Test() {
  const navigate = useNavigate();

  const percentage = 30;

  return (

    <>
      <div className="size-32 border border-black relative">
        <div className="w-full bg-blue-300 absolute bottom-0"
          style={{ height: `${percentage}%` }} // 자식 높이 비율
        > </div>
      </div>


      <div className="wave-container ">
        <svg
          viewBox="0 0 300 100"
          xmlns="http://www.w3.org/2000/svg"
          className="wave"
        >
          <path
            className="wave-path "
            d="M 0 50 Q 75 20, 150 50 T 300 50 V 100 H 0 Z"
            fill="lightblue"
          />
        </svg>
      </div>

      <div className="size-10 bg-lime-50">
        <WaterDrop/>
      </div>
    </>

  )
}
