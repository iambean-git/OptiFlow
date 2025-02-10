import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function NotLogined() {
    const navigate = useNavigate();

    useEffect(()=>{
        alert("로그인이 필요합니다.");
        navigate("/login");
    },[]);

  return (
    <div>
      
    </div>
  )
}
