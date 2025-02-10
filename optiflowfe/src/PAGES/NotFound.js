import { FaArrowRightLong } from "react-icons/fa6";
import { useNavigate } from "react-router-dom";

export default function NotFound() {
    const navigate = useNavigate();

    return (
        <div className="w-full h-screen flex justify-center items-center">
            <div className="w-full lg:w-2/3 h-full flex flex-col justify-center items-center ">
                <img src="/images/notFoundImg.png" className="w-[550px]" alt="404"></img>
                <div className="flex flex-col items-center mt-4">
                    <p className="text-6xl font-semibold">NOT FOUND</p>
                    <p className="text-2xl py-2">죄송합니다. 현재 찾을 수 없는 페이지를 요청 하셨습니다.</p>
                    <p className="text-gray-500">페이지의 주소가 잘못 입력되었거나</p>
                    <p className="text-gray-500">주소가 변경 혹은 요청하신 페이지를 찾을 수 없습니다..</p>
                    <div className="text-gray-500 mt-10 pl-5 pr-[14px] py-2
                                border border-gray-400 rounded-2xl 
                                hover:cursor-pointer"
                         onClick={()=>{navigate("/")}}
                    >
                        메인으로 이동 <span className="ml-6">→</span>

                    </div>
                </div>
            </div>

        </div>
    )
}
