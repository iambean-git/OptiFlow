import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
export default function Main() {
    const navigate = useNavigate();
    const loginId = null;
    // const loginId = "opti1";

    useEffect(()=>{
        if(loginId){
            navigate("/dashboard");
        }
    },[]);

  return (
    <div className="w-full h-screen">

      <div>
        main
      </div>
    </div>
  )
}
