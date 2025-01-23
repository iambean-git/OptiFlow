import { FaChevronRight } from "react-icons/fa6";
import { MdOutlineSpaceDashboard } from "react-icons/md";
import { useNavigate } from "react-router-dom";
import "./navBar/navbar.css";

import LocalTime from "./navBar/LocalTime";

export default function NavBar() {

    const navigate = useNavigate();
    
    const username = sessionStorage.getItem("username");

    const handleLogout = () => {
        if(window.confirm("로그아웃 하시겠습니까?")){
            sessionStorage.clear();
            navigate("/");
        }
    }
    return (
        <div className="navBG relative px-4 text-white">
            <header className="w-full px-2 py-4 border-b border-b-white flex justify-between items-center">
                <div className="">{username}님</div>
                <div className="border rounded-md text-xs p-2 hover:cursor-pointer"
                    onClick={handleLogout}>로그아웃</div>
            </header>

            {/* 날짜 및 시간 */}
            <div className="w-full flex flex-col justify-center items-center 
                            py-6 px-4">
                <LocalTime/>
                <div className="my-4"> 4℃ 맑음 </div>

            </div>


            {/* 메뉴 영역 */}
            <div className="w-[230px] h-[50px] flex items-center relative hover:cursor-pointer"
                onClick={()=>navigate("/dashboard")}>
                <MdOutlineSpaceDashboard className="size-6 mr-2"/>
                <p>대시보드</p>
                <FaChevronRight className="absolute right-2" />
            </div>
            
            <div className="w-[230px] h-[50px] flex items-center relative hover:cursor-pointer">
                <MdOutlineSpaceDashboard className="size-6 mr-2"/>
                <p>대시보드</p>
                <FaChevronRight className="absolute right-2" />
            </div>

            <div className="w-[230px] h-[50px] flex items-center relative hover:cursor-pointer" 
                onClick={()=>navigate("/regions")}>
                <MdOutlineSpaceDashboard className="size-6 mr-2"/>
                <p>지역별</p>
                <FaChevronRight className="absolute right-2" />
            </div>

            <img src="/images/logo_square_white.png" className="absolute bottom-6 left-0 px-12"></img>
        </div>
    )
}
